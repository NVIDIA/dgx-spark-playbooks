#/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
cp -r ~/notebooks/playbook/setup ~/notebooks/setup/
cd /home/rapids/notebooks/setup

# Install cuOpt kernel
echo "Installing cuOpt Portfolio Optimization Kernel."
python -m venv .venv
source .venv/bin/activate

# Install with all dependencies using uv for CUDA 13 (DGX SPark CUDA version)
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra cuda13 --locked --no-dev

# Install Jupyter and JupyterLab
uv pip install ipykernel

# Create a Jupyter kernel for this environment
uv run python -m ipykernel install --user --name=portfolio-opt --display-name "Portfolio Optimization"

# Copy necessaru playbook files
cp -r ~/notebooks/playbook/assets ~/notebooks/assets
cp ~/notebooks/playbook/README.md ~/notebooks/START_HERE.md
cp ~/notebooks/playbook/cvar_basic.ipynb ~/notebooks/cvar_basic.ipynb

set -m
# Start the primary process and put it in the background
jupyter-lab --notebook-dir=/home/rapids/notebooks --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*'