#!/usr/bin/env bash
# Deploy the Fair Store (CKAN 2.11 + ckanext-hierarchy) next to Verikan.
#
# One Compute Engine VM runs the same stack as the local `fairstore` compose
# profile — CKAN, Postgres, Solr, Redis — plus Caddy for automatic HTTPS. A VM
# rather than Cloud Run because Postgres and Solr need persistent disks.
#
# Idempotent: creates the static IP, firewall rule, VM and daily disk snapshots
# on the first run, then (re)ships docker/fairstore + this directory and
# restarts the stack. Secrets are generated on the VM and stay there.
#
#   gcloud auth login              # the session expires; re-auth is interactive
#   PROJECT=<gcp-project-id> deploy/fairstore/deploy.sh
#
# The VM runs without a service account: the site calls no Google API, and the
# default one is a project Editor. A VM that still has one is stopped, stripped
# of it and started again — a few minutes of downtime, once.
#
# The host is FAIRSTORE_HOST if set, else the one the VM already serves (its
# .env), else <ip>.sslip.io, a wildcard DNS name that resolves to the VM so
# Caddy can get a certificate before a real domain exists. Point a DNS name at
# the IP and rerun with FAIRSTORE_HOST set to move it.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the GCP project id to deploy into}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-verikan-fairstore}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
DISK_SIZE="${DISK_SIZE:-50GB}"
NETWORK_TAG="${VM}-web"
REMOTE_DIR=/opt/fairstore
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STARTUP_SCRIPT="$ROOT/deploy/fairstore/vm-startup.sh"
# Outside the repository, so the mirror's sysadmin token can never be committed.
TOKEN_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/verikan/fairstore-token"

gc() { gcloud --project "$PROJECT" --quiet "$@"; }
on_vm() { gc compute ssh "$VM" --zone "$ZONE" --command "$1"; }
vm_field() { gc compute instances describe "$VM" --zone "$ZONE" --format="value($1)"; }

# wait_for <timeout seconds> <seconds between tries> <what> <command...>
# Prints a dot per failed try and, on timeout, the last try's output (the why).
wait_for() {
    local limit="$1" pause="$2" what="$3" deadline out
    shift 3
    deadline=$((SECONDS + limit))
    printf 'Waiting for %s ' "$what"
    until out="$("$@" 2>&1)"; do
        if ((SECONDS >= deadline)); then
            echo
            echo "deploy: gave up waiting for $what after $((limit / 60)) min; last try said:" >&2
            printf '%s\n' "$out" | tail -n 20 >&2
            exit 1
        fi
        printf .
        sleep "$pause"
    done
    echo " ok"
}

gc projects describe "$PROJECT" --format='value(projectId)' >/dev/null \
    || { echo "gcloud cannot reach $PROJECT; run: gcloud auth login" >&2; exit 1; }

if ! gc compute addresses describe "${VM}-ip" --region "$REGION" >/dev/null 2>&1; then
    gc compute addresses create "${VM}-ip" --region "$REGION"
fi
IP="$(gc compute addresses describe "${VM}-ip" --region "$REGION" --format='value(address)')"

if ! gc compute firewall-rules describe "${VM}-web" >/dev/null 2>&1; then
    gc compute firewall-rules create "${VM}-web" --network default --direction INGRESS \
        --allow tcp:80,tcp:443 --target-tags "$NETWORK_TAG" --source-ranges 0.0.0.0/0
fi

if ! gc compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; then
    gc compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE_TYPE" \
        --image-family debian-12 --image-project debian-cloud \
        --boot-disk-size "$DISK_SIZE" --boot-disk-type pd-balanced \
        --address "$IP" --tags "$NETWORK_TAG" \
        --no-service-account --no-scopes --deletion-protection \
        --labels app=verikan,component=fairstore \
        --metadata-from-file startup-script="$STARTUP_SCRIPT"
else
    status="$(vm_field status)"
    sa="$(vm_field 'serviceAccounts[].email')"
    if [ -n "$sa" ]; then
        echo "$VM runs as $sa, which the site never uses; removing it."
        echo "That needs the VM stopped: the Fair Store is down for a few minutes."
        [ "$status" = TERMINATED ] || gc compute instances stop "$VM" --zone "$ZONE"
        gc compute instances set-service-account "$VM" --zone "$ZONE" \
            --no-service-account --no-scopes
        status=TERMINATED
    fi
    # The boot disk holds every database: guard the VM against deletion.
    gc compute instances update "$VM" --zone "$ZONE" --deletion-protection
    # GCE runs the startup script stored in instance metadata, so ship the
    # current one; it applies from the next boot (remote-up.sh covers this one).
    gc compute instances add-metadata "$VM" --zone "$ZONE" \
        --metadata-from-file startup-script="$STARTUP_SCRIPT"
    if [ "$status" = TERMINATED ]; then
        echo "Starting $VM..."
        gc compute instances start "$VM" --zone "$ZONE"
    fi
fi

# Postgres, Solr and uploads all live on the boot disk: keep a week of daily
# snapshots.
if ! gc compute resource-policies describe "${VM}-daily" --region "$REGION" >/dev/null 2>&1; then
    gc compute resource-policies create snapshot-schedule "${VM}-daily" --region "$REGION" \
        --daily-schedule --start-time 07:00 --max-retention-days 7 \
        --on-source-disk-delete keep-auto-snapshots
fi
policies="$(gc compute disks describe "$VM" --zone "$ZONE" --format='value(resourcePolicies)')"
case ";${policies};" in
    *"/resourcePolicies/${VM}-daily;"*) ;;
    *) gc compute disks add-resource-policies "$VM" --zone "$ZONE" --resource-policies "${VM}-daily" ;;
esac

# `compose ls` needs both the compose plugin and a running daemon.
wait_for 900 15 "Docker on $VM" on_vm "sudo docker compose ls"

# Default to the host the VM already serves: falling back to sslip.io on every
# run would silently move a site that has been given a real domain.
if [ -n "${FAIRSTORE_HOST:-}" ]; then
    HOST="$FAIRSTORE_HOST"
else
    HOST="$(on_vm "if sudo test -f $REMOTE_DIR/.env; then sudo sed -n 's/^FAIRSTORE_HOST=//p' $REMOTE_DIR/.env; fi" \
        | tail -n 1 | tr -d '\r')"
    HOST="${HOST:-${IP//./-}.sslip.io}"
fi
[[ "$HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "deploy: bad host name: '$HOST'" >&2; exit 1; }
echo "Deploying to https://$HOST"

bundle="$(mktemp -d)"
trap 'rm -rf "$bundle"' EXIT
mkdir -p "$bundle/stack/ckan"
cp "$ROOT"/deploy/fairstore/{docker-compose.yml,remote-up.sh} "$bundle/stack/"
cp -R "$ROOT"/deploy/fairstore/caddy "$bundle/stack/"
cp -R "$ROOT"/docker/fairstore/. "$bundle/stack/ckan/"
# macOS tar would add AppleDouble ._* files and xattrs, and record the local
# user as owner. Members are named rather than ".", so extracting never resets
# the mode of /opt/fairstore itself.
COPYFILE_DISABLE=1 tar --no-xattrs --owner=root:0 --group=root:0 --exclude .DS_Store \
    -C "$bundle/stack" -czf "$bundle/fairstore.tgz" docker-compose.yml remote-up.sh caddy ckan
gc compute scp "$bundle/fairstore.tgz" "$VM:/tmp/fairstore.tgz" --zone "$ZONE"
# ckan/ is only a build context (its init-db.sh runs only when Postgres
# initialises an empty volume), so it is replaced whole and files deleted here
# disappear there. .env is never touched, and neither is the caddy/ directory
# itself: Caddy bind-mounts it, so only the files inside it are replaced.
# The first deploys' bundles left macOS ._* files and the operator's ownership
# at the top level; both are tidied here.
on_vm "sudo mkdir -p $REMOTE_DIR && sudo chown root:root $REMOTE_DIR \
    && sudo find $REMOTE_DIR -maxdepth 1 -name '._*' -delete \
    && sudo rm -rf $REMOTE_DIR/ckan \
    && sudo tar --no-same-owner -xzf /tmp/fairstore.tgz -C $REMOTE_DIR \
    && rm -f /tmp/fairstore.tgz \
    && sudo bash $REMOTE_DIR/remote-up.sh $HOST"

wait_for 600 10 "https://$HOST" \
    curl -fsS -o /dev/null --max-time 20 "https://$HOST/api/3/action/status_show"
cat <<EOF

Fair Store is up: https://$HOST

Mint an API token for the mirror into a private file outside the repository:
  (umask 077 && mkdir -p "${TOKEN_FILE%/*}" && gcloud compute ssh $VM --zone $ZONE --project $PROJECT --command \\
    "cd $REMOTE_DIR && sudo docker compose exec -T fairstore ckan -c /srv/app/ckan.ini user token add fairadmin mirror" \\
    | tail -1 > "$TOKEN_FILE")
then load every registered portal:
  python -m scripts.populate_fairstore --site all --apply \\
    --target-url https://$HOST --api-key-file "$TOKEN_FILE"

The fairadmin password is FAIRSTORE_ADMIN_PASSWORD in $REMOTE_DIR/.env on the VM.
EOF
