#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
CURR_DIR="$(pwd -P)"
if [[ "$CURR_DIR" != "$SCRIPT_DIR" ]]; then
    echo "Error: Please run this script from its own directory: $SCRIPT_DIR"
    exit 1
fi

if [[ "$EUID" -eq 0 ]]; then
    echo "Error: This script must not be run as root."
    exit 1
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 --help to see the available options"
    exit 1
fi

if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "---- Installing required packages ----"
pip install -r requirements.txt

echo "---- Configuring the cluster (args: $*) ----"
SPARK_CLUSTER_SETUP_WRAPPER=1 python3 ./spark_cluster_setup.py "$@"

deactivate
