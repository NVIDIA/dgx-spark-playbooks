FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /workspace

# Install dependencies globally so torchrun (which uses /usr/bin/python) can access them
RUN /usr/bin/python -m pip install tiktoken tokenizers datasets psutil files-to-prompt regex setuptools uvicorn wandb maturin

# Create venv with --system-site-packages so it inherits global packages
RUN /usr/bin/python -m venv --system-site-packages .venv

CMD ["/bin/bash"]