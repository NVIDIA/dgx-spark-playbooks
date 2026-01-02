#!/bin/bash

# Get the script's directory, handling symlinks and different invocation methods
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PARENT_DIR="${SCRIPT_DIR%/*}"
# Print the location
echo "Starting container with volumized data folder at: $PARENT_DIR"

docker run --gpus all --pull always --rm -it     --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864    -p 8888:8888 -p 8787:8787 -p 8786:8786 -p 8501:8501 -p 8050:8050     -v $PARENT_DIR:/home/rapids/notebooks/playbook     nvcr.io/nvidia/rapidsai/notebooks:25.10-cuda13-py3.13 bash /home/rapids/notebooks/playbook/setup/setup_playbook.sh