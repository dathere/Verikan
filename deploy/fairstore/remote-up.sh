#!/usr/bin/env bash
# Runs on the Fair Store VM, as root, from deploy.sh: create the secrets on the
# first deploy, then build and (re)start the stack.
#
#   remote-up.sh <public hostname>
#
# .env is generated once and never leaves the VM. Rotating a value means
# editing .env; note the database passwords are only applied when Postgres
# initialises an empty volume.
set -euo pipefail
cd "$(dirname "$0")"
host="${1:?usage: remote-up.sh <hostname>}"
[[ "$host" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "remote-up: bad hostname: $host" >&2; exit 1; }

# retry <tries> <seconds between tries> <command...>
retry() {
    local tries="$1" pause="$2" i
    shift 2
    for ((i = 1; ; i++)); do
        "$@" && return 0
        ((i < tries)) || return 1
        sleep "$pause"
    done
}

# Containers get no path to the metadata server, DNS (port 53) excepted; see
# vm-startup.sh, which re-adds these on every boot. Added here too so a deploy
# takes effect without a reboot.
for proto in tcp udp; do
    iptables -C DOCKER-USER -d 169.254.169.254 -p "$proto" ! --dport 53 -j DROP 2>/dev/null \
        || iptables -I DOCKER-USER -d 169.254.169.254 -p "$proto" ! --dport 53 -j DROP
done

if [ ! -f .env ]; then
    umask 077
    secret() { openssl rand -hex 24; }
    {
        echo "FAIRSTORE_HOST=${host}"
        echo "FAIRSTORE_SECRET_KEY=$(secret)"
        echo "FAIRSTORE_POSTGRES_PASSWORD=$(secret)"
        echo "FAIRSTORE_DB_PASSWORD=$(secret)"
        echo "FAIRSTORE_DATASTORE_PASSWORD=$(secret)"
        echo "FAIRSTORE_ADMIN_PASSWORD=$(secret)"
    } > .env
fi
# Solr keeps each dataset as indexed, with absolute URLs built from the site
# URL, so a host change needs a reindex. The marker survives a failed run, so
# the next deploy still does it.
if [ "$(sed -n 's/^FAIRSTORE_HOST=//p' .env | tail -n 1)" != "$host" ]; then
    touch .reindex-pending
fi
sed -i "s|^FAIRSTORE_HOST=.*|FAIRSTORE_HOST=${host}|" .env

docker compose up -d --build --remove-orphans
# Caddy reads its config only at start; reload applies a changed Caddyfile
# without dropping connections. Retried because a just-created container may
# not have its admin endpoint up yet.
retry 10 3 docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile \
    || { echo "remote-up: caddy reload failed (the previous config is still live)" >&2; exit 1; }
# Before the caddy/ directory mount the Caddyfile sat here; nothing reads it now.
rm -f Caddyfile

if [ -f .reindex-pending ]; then
    ckan_up() {
        docker compose exec -T fairstore python3 -c \
            'import urllib.request as u; u.urlopen("http://127.0.0.1:5000/api/3/action/status_show", timeout=10)' \
            >/dev/null 2>&1
    }
    echo "Site URL is now https://${host}: waiting for CKAN, then rebuilding the search index..."
    retry 60 5 ckan_up \
        || { echo "remote-up: CKAN did not answer after 60 tries; search index NOT rebuilt (the next deploy retries)" >&2; exit 1; }
    docker compose exec -T fairstore ckan -c /srv/app/ckan.ini search-index rebuild
    rm -f .reindex-pending
fi
