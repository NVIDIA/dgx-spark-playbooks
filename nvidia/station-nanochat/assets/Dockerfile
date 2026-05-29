FROM nvcr.io/nvidia/pytorch:26.04-py3

WORKDIR /workspace

RUN pip install \
    datasets \
    tokenizers \
    wandb \
    tiktoken \
    psutil \
    files-to-prompt \
    uvicorn \
    rustbpe

CMD ["/bin/bash"]