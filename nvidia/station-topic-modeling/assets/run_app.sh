#!/usr/bin/env bash
# Launch the interactive BERTopic dashboard.
#
# Uses the rapids-25.10 conda env (only one with streamlit + cuml.accel +
# bertopic + datamapplot all installed). Override with PYTHON env var if needed.
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Interpreter resolution (override with PYTHON=...):
#   1. explicit $PYTHON
#   2. the rapids-25.10 conda env, if conda can find it
#   3. whatever `python` is on PATH (e.g. an already-activated env)
ENV_NAME="${CONDA_ENV:-rapids-25.10}"
if [[ -z "${PYTHON:-}" ]]; then
    if command -v conda >/dev/null 2>&1 && \
       PYTHON="$(conda run -n "$ENV_NAME" which python 2>/dev/null)" && \
       [[ -n "$PYTHON" ]]; then
        :
    else
        PYTHON="$(command -v python)"
    fi
fi

if [[ -z "$PYTHON" ]]; then
    echo "error: no python interpreter found; set PYTHON=/path/to/python" >&2
    exit 1
fi

exec "$PYTHON" -m streamlit run topic_modeling_app.py \
    --server.address 0.0.0.0 \
    --server.port "${PORT:-8501}" \
    --server.headless true
