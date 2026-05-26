#!/usr/bin/env bash
# Train the LightGBM lambdarank re-ranker on cached HLLM embeddings.
#
# Architecture: group-by-user lambdarank over FAISS top-100 candidates with
# ~21 handcrafted features.

set -euo pipefail

WORKSPACE="${PLAYBOOK_WORKSPACE:-$HOME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROCESSED_DIR="${PLAYBOOK_PROCESSED_DIR:-$WORKSPACE/data/processed}"
OUTPUT_DIR="${PLAYBOOK_RERANKER_DIR:-$WORKSPACE/models/reranker_lightgbm}"

exec uv run --project "$REPO_DIR" python "$SCRIPT_DIR/train_reranker_lightgbm.py" \
    --processed-dir "$PROCESSED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
