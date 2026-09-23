#!/bin/bash
# Compute Engine startup script for the Fair Store VM (runs as root on every
# boot). Installs Docker Engine and the compose plugin on first boot, then, on
# every boot, keeps containers away from the metadata server. The stack itself
# restarts on its own (restart: unless-stopped).
#
# GCE reads this from instance metadata, not from the repository: deploy.sh
# re-uploads it on each run, and a change takes effect at the next boot.
set -euo pipefail

install_docker() {
    # A first boot races GCE's own apt runs (unattended-upgrades, the guest
    # agent), so wait for the lock rather than fail.
    local apt=(apt-get -o DPkg::Lock::Timeout=600)
    "${apt[@]}" update
    "${apt[@]}" install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" \
        > /etc/apt/sources.list.d/docker.list
    "${apt[@]}" update
    "${apt[@]}" install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable --now docker
}

# The compose plugin is the last package the stack needs: a first boot cut
# short after the docker CLI was unpacked is finished on the next boot.
docker compose version >/dev/null 2>&1 || install_docker

# Nothing in the stack calls a Google API, so containers get no path to the
# metadata server (the VM has no service account either; this is the second
# layer). Port 53 stays open: on GCE that address is also the VM's DNS
# resolver, which the image build's RUN steps and any default-bridge
# container query directly (containers on the compose network go through
# Docker's embedded resolver), so a blanket DROP would break image builds.
# iptables rules do not survive a reboot, and DOCKER-USER only exists once
# dockerd has started. remote-up.sh adds the same rules on each deploy.
for _ in $(seq 60); do
    iptables -n -L DOCKER-USER >/dev/null 2>&1 && break
    sleep 5
done
if ! iptables -n -L DOCKER-USER >/dev/null 2>&1; then
    echo "vm-startup: no DOCKER-USER chain after 5 min (is dockerd running?);" \
        "containers can still reach the metadata server" >&2
    exit 1
fi
for proto in tcp udp; do
    iptables -C DOCKER-USER -d 169.254.169.254 -p "$proto" ! --dport 53 -j DROP 2>/dev/null \
        || iptables -I DOCKER-USER -d 169.254.169.254 -p "$proto" ! --dport 53 -j DROP
done
