#!/usr/bin/env bash
# Launch the Wan2GP (WanGP) web UI on a DGX Spark, exposed on the LAN.
# Open it from another machine at  http://<spark-hostname>.local:7860
set -euo pipefail

ENV_DIR="${WAN2GP_ENV_DIR:-$HOME/wan2gp-env}"
REPO_DIR="${WAN2GP_REPO_DIR:-$HOME/Wan2GP}"

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
cd "$REPO_DIR"

# --listen binds Gradio to 0.0.0.0 so the UI is reachable from the LAN.
# Run `python wgp.py --help` to see all options (port, profiles, etc.).
python wgp.py --listen "$@"
