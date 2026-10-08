#!/bin/sh
set -eu
root=/opt/livinghome-validation
mkdir -p "$root/homeassistant"
if [ -e "$root/homeassistant/configuration.yaml" ]; then
  echo 'The lab configuration already exists. Refusing to overwrite it.' >&2
  exit 2
fi
cp /mnt/c/LivingHome-installer/validation/ha-configuration.yaml "$root/homeassistant/configuration.yaml"
printf '[]\n' > "$root/homeassistant/automations.yaml"
podman --cgroup-manager=cgroupfs run -d --name livinghome-validation-ha \
  --cgroups=disabled --restart=no \
  -p 127.0.0.1:18123:8123 \
  -v "$root/homeassistant:/config" \
  -e TZ=America/Los_Angeles \
  ghcr.io/home-assistant/home-assistant:2026.9.4
