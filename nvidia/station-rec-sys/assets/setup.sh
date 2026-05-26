#!/usr/bin/env bash
# setup.sh — Automated setup for the standalone recommender playbook.
#
# Installs all dependencies, downloads models and data, and prepares the
# environment to run the E2E HLLM recommendation pipeline.
# Idempotent: safe to re-run if interrupted — skips completed steps.
#
# USAGE EXAMPLES:
#
#   # Default setup — stores data/models under $HOME
#   bash setup.sh
#
#   # Preview what would be done without executing
#   bash setup.sh --dry-run
#
#   # Verify prerequisites only
#   bash setup.sh --check
#
#   # Store large files on a different mount (SSD, NVMe, RAID array, etc.)
#   bash setup.sh --workspace /raid/recsys-playbook
#   bash setup.sh --workspace /mnt/nvme/recsys-playbook
#   bash setup.sh --workspace /data/recsys-playbook
#   bash setup.sh --data-dir /raid/recsys-playbook
#
#   # Use environment variable instead of CLI flag
#   export PLAYBOOK_WORKSPACE=/raid
#   bash setup.sh
#
#   # Show all options
#   bash setup.sh --help
#
# WHAT IT INSTALLS under $PLAYBOOK_WORKSPACE (~80 GB total):
#   - uv (Python project manager)        — ~/.local/bin/
#   - Ollama (LLM serving)               — /usr/local/bin/ (requires sudo)
#   - Nemotron Mini 4B (Ollama)          — ~/.ollama/models/ (~3 GB)
#   - Python deps (uv venv)              — this repo's .venv/
#   - HLLM code (ByteDance)              — $WORKSPACE/hllm-code/ (~1 GB)
#   - TinyLlama-1.1B model               — $WORKSPACE/models/ (~2 GB, HF cache stays at default)
#   - Amazon Clothing dataset            — $WORKSPACE/data/ (~46 GB)
#   - Training checkpoints               — $WORKSPACE/checkpoints/ (~20–30 GB after training)
#
# REQUIRES: NVIDIA GPU with drivers installed, internet access.
# SUDO: Required only for Ollama install (can be pre-installed by an admin).

set -euo pipefail

DRY_RUN=false
CHECK_ONLY=false
WORKSPACE="${PLAYBOOK_WORKSPACE:-$HOME}"

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --check)        CHECK_ONLY=true ;;
        --workspace=*)  WORKSPACE="${arg#*=}" ;;
        --data-dir=*)   WORKSPACE="${arg#*=}" ;;
        --workspace)    ;; # handled below with next arg
        --data-dir)     ;; # handled below with next arg
        -h|--help)
            cat <<'HELPEOF'
Usage: bash setup.sh [OPTIONS]

Options:
  (no flags)        Run full setup
  --dry-run         Print commands without executing
  --check           Validate prerequisites only
  --workspace PATH  Root directory for all playbook artifacts
                    (default: $HOME, or $PLAYBOOK_WORKSPACE if set)
                    Example: bash setup.sh --workspace /raid/recsys-playbook
  --data-dir PATH   Alias for --workspace
  -h, --help        Show this help

Environment:
  PLAYBOOK_WORKSPACE   Same as --workspace (CLI flag takes precedence)

What gets created under WORKSPACE (~80 GB total):
  WORKSPACE/
  ├── station-rec-sys/          The playbook repo + uv venv
  ├── hllm-code/                ByteDance HLLM (~1 GB)
  ├── data/                     Amazon dataset — raw + processed (~46 GB)
  ├── models/                   TinyLlama-1.1B (~2 GB)
  └── checkpoints/              HLLM training output (~20–30 GB after training)

HuggingFace and Ollama caches stay at their default locations (~/.cache/huggingface,
~/.ollama) so they can be shared with other projects on the machine.
HELPEOF
            exit 0
            ;;
        *)
            if [[ "${prev_arg:-}" == "--workspace" ]]; then
                WORKSPACE="$arg"
            elif [[ "${prev_arg:-}" == "--data-dir" ]]; then
                WORKSPACE="$arg"
            else
                echo "Unknown option: $arg (run --help for usage)"
                exit 1
            fi
            ;;
    esac
    prev_arg="$arg"
done

WORKSPACE="${WORKSPACE%/}"   # strip trailing slash if any
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HLLM_CODE_DIR="$WORKSPACE/hllm-code"
DATA_DIR="$WORKSPACE/data"
MODELS_DIR="$WORKSPACE/models"
CHECKPOINTS_DIR="$WORKSPACE/checkpoints"

if $CHECK_ONLY; then
    echo "Checking pre-requisites..."
    echo ""

    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    RED='\033[0;31m'
    NC='\033[0m'

    ok=0
    warn=0
    fail=0

    if command -v nvidia-smi &>/dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        gpu_mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        gpu_mem_gb=$(( gpu_mem_mib / 1024 ))
        echo -e "  ${GREEN}[OK]${NC} GPU: $gpu_name (${gpu_mem_gb} GB)"
        ok=$((ok + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} nvidia-smi not found — NVIDIA driver not installed"
        fail=$((fail + 1))
    fi

    if command -v nvcc &>/dev/null; then
        cuda_ver=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        echo -e "  ${GREEN}[OK]${NC} CUDA: $cuda_ver"
        ok=$((ok + 1))
    else
        echo -e "  ${YELLOW}[WARN]${NC} nvcc not found — CUDA toolkit not on PATH (driver CUDA may still work)"
        warn=$((warn + 1))
    fi

    avail_gb=$(df -BG "$WORKSPACE" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
    if [ "${avail_gb:-0}" -ge 80 ]; then
        echo -e "  ${GREEN}[OK]${NC} Disk: ${avail_gb} GB available at $WORKSPACE (need ~80 GB)"
        ok=$((ok + 1))
    else
        echo -e "  ${YELLOW}[WARN]${NC} Disk: ${avail_gb:-?} GB available at $WORKSPACE (need ~80 GB)"
        echo "         Set PLAYBOOK_WORKSPACE to a path with more space, or use --workspace"
        warn=$((warn + 1))
    fi

    if command -v sudo &>/dev/null && sudo -n true 2>/dev/null; then
        echo -e "  ${GREEN}[OK]${NC} sudo: available (needed for Ollama install)"
        ok=$((ok + 1))
    else
        echo -e "  ${YELLOW}[WARN]${NC} sudo: not available or requires password"
        echo "         Ollama install needs sudo — ask an admin to pre-install if needed"
        warn=$((warn + 1))
    fi

    for tool in git wget curl; do
        if command -v "$tool" &>/dev/null; then
            echo -e "  ${GREEN}[OK]${NC} $tool: $(command -v "$tool")"
            ok=$((ok + 1))
        else
            echo -e "  ${RED}[FAIL]${NC} $tool: not found"
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo -e "Result: ${GREEN}$ok passed${NC}, ${YELLOW}$warn warnings${NC}, ${RED}$fail failed${NC}"
    if [ "$fail" -gt 0 ]; then
        echo "Fix failures above before running setup."
    elif [ "$warn" -gt 0 ]; then
        echo "Review warnings above before running setup."
    else
        echo "All checks passed. Ready to run: bash assets/setup.sh"
    fi
    exit 0
fi

run() {
    echo "  -> $*"
    $DRY_RUN || "$@"
}

section() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

echo "Workspace:    $WORKSPACE"
echo "Repo:         $REPO_DIR"

# ---------------------------------------------------------------
section "Step 1: System tools (uv, Ollama)"
# ---------------------------------------------------------------

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    run bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "uv already installed: $(uv --version)"
fi

if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    run bash -c 'curl -fsSL https://ollama.com/install.sh | sh'
else
    echo "Ollama already installed: $(ollama --version)"
fi

# ---------------------------------------------------------------
section "Step 2: Python environment (uv)"
# ---------------------------------------------------------------

cd "$REPO_DIR"

if [ -d ".venv" ]; then
    echo "Project venv already exists. Syncing dependencies..."
else
    echo "Creating project venv and installing all dependencies..."
fi
run uv sync --inexact

echo ""
echo "Verifying key packages..."
CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA: {torch.version.cuda}, GPU available: {torch.cuda.is_available()}')" || echo "  WARNING: PyTorch import failed"
uv run python -c "import faiss; print(f'  FAISS installed')" || echo "  WARNING: faiss not found"
uv run python -c "import transformers; print(f'  transformers {transformers.__version__}')" || echo "  WARNING: transformers import failed"
CUDA_VISIBLE_DEVICES=0 uv run python -c "import deepspeed; print(f'  deepspeed {deepspeed.__version__}')" || echo "  WARNING: deepspeed import failed"
uv run python -c "import peft; print(f'  peft {peft.__version__}')" || echo "  WARNING: peft import failed"

echo ""
echo "Optional W&B login..."
if uv run wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "  W&B already logged in."
elif [ -t 0 ]; then
    read -r -p "  Log in to Weights & Biases now? [y/N] " wandb_confirm
    if [[ "$wandb_confirm" =~ ^[Yy]$ ]]; then
        run uv run wandb login
    else
        echo "  Skipping W&B login. Run 'uv run wandb login' later to enable training dashboards."
    fi
else
    echo "  Non-interactive shell; skipping W&B login prompt."
    echo "  Run 'uv run wandb login' later to enable training dashboards."
fi

# ---------------------------------------------------------------
section "Step 3: flash-attn (source build for target GPU)"
# ---------------------------------------------------------------

# flash-attn must be built from source to target specific GPU architectures.
# Pre-built wheels only cover older CUDA versions.
# Note: uv sync --inexact (Step 2) preserves the editable install, so
# re-runs skip this step entirely once flash-attn is built.
FLASH_ATTN_REPO="$HOME/dev/flash-attention"
FLASH_ATTN_PY="$REPO_DIR/.venv/bin/python"
FLASH_ATTN_EDITABLE_WHEEL="${FLASH_ATTN_EDITABLE_WHEEL:-}"

if uv run python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed: $(uv run python -c 'import flash_attn; print(flash_attn.__version__)')"
else
    if [ -f "$FLASH_ATTN_REPO/flash_attn_2_cuda.cpython-313-aarch64-linux-gnu.so" ]; then
        echo "Found existing flash-attn source build at $FLASH_ATTN_REPO"
        if [ -z "$FLASH_ATTN_EDITABLE_WHEEL" ]; then
            FLASH_ATTN_EDITABLE_WHEEL=$(find "$HOME/.cache/uv/sdists-v9/editable" \
                -path "*/flash_attn-2.8.4-0.editable-cp313-cp313-linux_aarch64.whl" \
                -print -quit 2>/dev/null || true)
        fi
        if [ -n "$FLASH_ATTN_EDITABLE_WHEEL" ] && [ -f "$FLASH_ATTN_EDITABLE_WHEEL" ]; then
            echo "  Installing editable flash-attn wheel without rebuilding:"
            echo "  $FLASH_ATTN_EDITABLE_WHEEL"
            run uv pip install --python "$FLASH_ATTN_PY" "$FLASH_ATTN_EDITABLE_WHEEL"
        else
            echo "  Existing compiled extension found, but no editable wheel was found in uv cache."
        fi
    fi

    if uv run python -c "import flash_attn, flash_attn_2_cuda; print(f'  flash-attn {flash_attn.__version__} from {flash_attn.__file__}')" 2>/dev/null; then
        echo "  Reused existing flash-attn source build."
    else
    # Full source compilation required (~20 min first time)
    # Detect CUDA toolkit — prefer 13.1, fall back to whatever /usr/local/cuda points to
    if [ -d "/usr/local/cuda-13.1" ]; then
        export CUDA_HOME="/usr/local/cuda-13.1"
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
    echo "  CUDA_HOME: ${CUDA_HOME:-not set}"

    # Target GPU architectures — auto-detect from nvidia-smi so we only compile
    # kernels for GPUs actually present. Two env vars matter:
    #   - TORCH_CUDA_ARCH_LIST: dot-notation (e.g. "10.3"), used by torch's C++
    #     extension builder. Accepts the raw compute_cap string.
    #   - FLASH_ATTN_CUDA_ARCHS: flash-attn's own var (setup.py:72). Semicolon-
    #     separated family integers (80;90;100;110;120). Ignores
    #     TORCH_CUDA_ARCH_LIST entirely, so it must be set separately.
    # Mapping: "10.3" -> major=10 -> "100" (the sm_100f family, forward-
    # compatible with sm_101/102/103).
    detected_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ';' | sed 's/;$//')
    if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
        export TORCH_CUDA_ARCH_LIST="${detected_cc:-10.3}"
    fi
    if [ -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
        fa_archs=$(echo "$detected_cc" | tr ';' '\n' | awk -F. '{print $1*10}' | sort -u | tr '\n' ';' | sed 's/;$//')
        export FLASH_ATTN_CUDA_ARCHS="${fa_archs:-100}"
    fi
    echo "  Detected compute_cap:   ${detected_cc:-unknown}"
    echo "  TORCH_CUDA_ARCH_LIST:   $TORCH_CUDA_ARCH_LIST  (for torch extensions)"
    echo "  FLASH_ATTN_CUDA_ARCHS:  $FLASH_ATTN_CUDA_ARCHS  (flash-attn family targets)"

    if [ -d "$FLASH_ATTN_REPO" ]; then
        echo "  Building flash-attn from local repo: $FLASH_ATTN_REPO"
        echo "  First-time build compiles CUDA kernels from source using all CPU cores."
        echo "  This may take up to 30 minutes. Subsequent runs are near-instant."
        cd "$FLASH_ATTN_REPO"
        run git pull
    else
        echo "  Cloning flash-attention repo and building from source..."
        echo "  First-time build compiles CUDA kernels from source using all CPU cores."
        echo "  This may take up to 30 minutes. Subsequent runs are near-instant."
        run mkdir -p "$HOME/dev"
        run git clone https://github.com/Dao-AILab/flash-attention.git "$FLASH_ATTN_REPO"
        cd "$FLASH_ATTN_REPO"
    fi

    # Install into the project venv (--python points uv to the right env)
    run uv pip install --python "$FLASH_ATTN_PY" -e . --no-build-isolation
    cd "$REPO_DIR"

    # Verify
    if uv run python -c "import flash_attn; print(f'  flash-attn {flash_attn.__version__}')" 2>/dev/null; then
        echo "  flash-attn installed successfully."
    else
        echo "  WARNING: flash-attn build failed. Training may still work without it (slower attention)."
    fi
    fi
fi

# ---------------------------------------------------------------
section "Step 4: Clone and patch HLLM"
# ---------------------------------------------------------------

if [ ! -d "$HLLM_CODE_DIR/REC" ]; then
    echo "Cloning HLLM from github.com/bytedance/HLLM..."
    run mkdir -p "$WORKSPACE"
    UPSTREAM_TMP="$WORKSPACE/.hllm-upstream"
    run git clone https://github.com/bytedance/HLLM.git "$UPSTREAM_TMP"
    run cp -r "$UPSTREAM_TMP/code" "$HLLM_CODE_DIR"
    run rm -rf "$UPSTREAM_TMP"
else
    echo "HLLM code already present at $HLLM_CODE_DIR/"
fi

# Apply LoRA patches (overwrites upstream files with patched versions)
PATCHES_DIR="$REPO_DIR/assets/patches/HLLM"
if [ -d "$PATCHES_DIR" ]; then
    echo "  Applying LoRA patches from $PATCHES_DIR ..."
    run cp "$PATCHES_DIR/hllm.py"            "$HLLM_CODE_DIR/REC/model/HLLM/hllm.py"
    run cp "$PATCHES_DIR/modeling_bert.py"   "$HLLM_CODE_DIR/REC/model/HLLM/modeling_bert.py"
    run cp "$PATCHES_DIR/trainer.py"         "$HLLM_CODE_DIR/REC/trainer/trainer.py"
    run cp "$PATCHES_DIR/wandblogger.py"     "$HLLM_CODE_DIR/REC/utils/wandblogger.py"
    run cp "$PATCHES_DIR/argument_list.py"   "$HLLM_CODE_DIR/REC/utils/argument_list.py"
    run cp "$PATCHES_DIR/utils.py"           "$HLLM_CODE_DIR/REC/data/utils.py"
    run cp "$PATCHES_DIR/dataload.py"        "$HLLM_CODE_DIR/REC/data/dataload.py"
    echo "  LoRA patches applied."
fi

# Dataset + information dirs live alongside the HLLM code (HLLM config reads from them)
run mkdir -p "$HLLM_CODE_DIR/dataset" "$HLLM_CODE_DIR/information" "$CHECKPOINTS_DIR"

# Copy training/extraction scripts from the repo into hllm-code/ for HLLM to find
if [ -f "$REPO_DIR/assets/train_retriever.sh" ]; then
    run cp "$REPO_DIR/assets/train_retriever.sh" "$HLLM_CODE_DIR/train_lora.sh"
    run chmod +x "$HLLM_CODE_DIR/train_lora.sh"
fi
if [ -f "$REPO_DIR/assets/extract_embeddings.py" ]; then
    run cp "$REPO_DIR/assets/extract_embeddings.py" "$HLLM_CODE_DIR/extract_embeddings.py"
fi

# Create extract_embeddings.sh wrapper inside hllm-code/
cat > "$HLLM_CODE_DIR/extract_embeddings.sh" << SCRIPT
#!/bin/bash
set -e
cd "$HLLM_CODE_DIR"
unset MPLBACKEND
CKPT_PATH="\${1:-$CHECKPOINTS_DIR/dresses_lora_r16/HLLM-0.pth/checkpoint/mp_rank_00_model_states.pt}"
OUTPUT_DIR="\${2:-$DATA_DIR/processed}"
MASTER_PORT="\${PLAYBOOK_EXTRACT_MASTER_PORT:-12399}"
[ "\$#" -ge 1 ] && shift
[ "\$#" -ge 1 ] && shift
exec uv run --project "$REPO_DIR" torchrun --master_port="\$MASTER_PORT" --nproc_per_node=1 --nnodes=1 extract_embeddings.py \\
    --ckpt_path "\$CKPT_PATH" \\
    --output_dir "\$OUTPUT_DIR" \\
    "\$@"
SCRIPT
chmod +x "$HLLM_CODE_DIR/extract_embeddings.sh"

# ---------------------------------------------------------------
section "Step 5: Download models"
# ---------------------------------------------------------------

# TinyLlama-1.1B (backbone for HLLM retriever training)
# If the download fails with 401/403, log in first: uv run hf login
if [ -f "$MODELS_DIR/TinyLlama-1.1B/config.json" ]; then
    echo "TinyLlama-1.1B already present at $MODELS_DIR/TinyLlama-1.1B/"
else
    echo "Downloading TinyLlama-1.1B from HuggingFace (~2 GB)..."
    echo "  If this fails with a 401/403 error, run: uv run hf login"
    run mkdir -p "$MODELS_DIR"
    run uv run hf download \
        TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --local-dir "$MODELS_DIR/TinyLlama-1.1B"
fi

# Nemotron Mini via Ollama
if ollama list 2>/dev/null | grep -q "nemotron-mini"; then
    echo "nemotron-mini already pulled in Ollama."
else
    echo "Pulling nemotron-mini (2.7 GB)..."
    run ollama pull nemotron-mini
fi

# ---------------------------------------------------------------
section "Step 6: Download and process Amazon data"
# ---------------------------------------------------------------

REVIEWS_FILE="$DATA_DIR/raw/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl"
META_FILE="$DATA_DIR/raw/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl"

run mkdir -p "$DATA_DIR/raw/raw/review_categories" "$DATA_DIR/raw/raw/meta_categories" "$DATA_DIR/processed"

# Dataset moved from datarepo.eng.ucsd.edu to HuggingFace (plain JSONL, not gzipped)
HF_DATASET_BASE="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

if [ -f "$REVIEWS_FILE" ]; then
    echo "Reviews data already present ($(du -h "$REVIEWS_FILE" | cut -f1))"
else
    echo "Downloading Amazon Reviews — Clothing, Shoes & Jewelry (~27.8 GB)..."
    run wget -c \
        "$HF_DATASET_BASE/raw/review_categories/Clothing_Shoes_and_Jewelry.jsonl" \
        -O "$REVIEWS_FILE"
fi

if [ -f "$META_FILE" ]; then
    echo "Metadata already present ($(du -h "$META_FILE" | cut -f1))"
else
    echo "Downloading Amazon Metadata — Clothing, Shoes & Jewelry (~18 GB)..."
    run wget -c \
        "$HF_DATASET_BASE/raw/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl" \
        -O "$META_FILE"
fi

# Process into HLLM format
if [ -f "$DATA_DIR/processed/dress_metadata.parquet" ] && [ -f "$HLLM_CODE_DIR/dataset/amazon_dresses.csv" ]; then
    echo "Processed data already exists. Skipping processing."
else
    echo "Processing Amazon data into HLLM format..."
    cd "$REPO_DIR"
    run uv run python assets/prepare_data.py
fi

# ---------------------------------------------------------------
section "Step 7: Start Ollama"
# ---------------------------------------------------------------

if curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Ollama is already running."
    ollama list
else
    echo "Starting Ollama..."
    run ollama serve &
    sleep 3
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "Ollama started successfully."
        ollama list
    else
        echo "WARNING: Ollama may not have started. Run 'ollama serve' manually."
    fi
fi

# ---------------------------------------------------------------
section "Setup complete!"
# ---------------------------------------------------------------

echo ""
echo "Workspace: $WORKSPACE"
echo "Repo:      $REPO_DIR"
echo ""
echo "Next steps:"
echo "  cd $REPO_DIR"
echo "  bash assets/train_retriever.sh"
echo "  uv run python assets/extract_embeddings.py"
echo ""
echo "Expected runtime: ~20 min (with training at bs=512) or ~5 min (with pre-computed embeddings)"
echo ""
