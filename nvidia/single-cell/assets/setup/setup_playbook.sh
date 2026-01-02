#/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

mamba install -c conda-forge compilers -y
cd /home/rapids/notebooks/playbook/setup
uv pip install --system -r requirements.txt

cp -r ~/notebooks/playbook/assets ~/notebooks/assets
cp ~/notebooks/playbook/README.md ~/notebooks/START_HERE.md
cp ~/notebooks/playbook/scRNA_analysis_preprocessing.ipynb ~/notebooks/scRNA_analysis_preprocessing.ipynb 

set -m
# Start the primary process and put it in the background
jupyter-lab --notebook-dir=/home/rapids/notebooks/ --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' 