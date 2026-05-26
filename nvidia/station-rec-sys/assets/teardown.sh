#!/usr/bin/env bash
# teardown.sh — Stop processes and optionally remove all downloaded assets.
#
# Kills running demo processes (Ollama, vLLM, training, FastAPI,
# W&B). With --purge-downloads, also removes all downloaded data,
# base models, environments, and HLLM code — but preserves trained checkpoints.
#
# USAGE EXAMPLES:
#
#   # Kill all running demo processes
#   bash teardown.sh
#
#   # Preview what would be killed (no action taken)
#   bash teardown.sh --dry-run
#
#   # Kill processes AND remove all downloaded assets (~60 GB)
#   bash teardown.sh --purge-downloads
#
#   # Same, but data lives on a custom mount
#   bash teardown.sh --purge-downloads --data-dir /raid
#
#   # Preview what --purge-downloads would remove
#   bash teardown.sh --purge-downloads --dry-run
#
#   # Use environment variable instead of CLI flag
#   export PLAYBOOK_WORKSPACE=/raid
#   bash teardown.sh --purge-downloads
#
#   # Show all options
#   bash teardown.sh --help
#
# WHAT --purge-downloads REMOVES:
#   - Amazon raw + processed data         (~46 GB)
#   - TinyLlama-1.1B base model           (~2 GB)
#   - Ollama nemotron-mini model           (~3 GB)
#   - HLLM code, dataset, information      (~1 GB)
#   - Project .venv, wandb/tensorboard logs (~2+ GB)
#
# WHAT --purge-downloads PRESERVES:
#   - Saved checkpoints ($WORKSPACE/checkpoints/) — your trained model weights
#   - This repo's source code
#   - uv and ollama binaries (system tools)

set -euo pipefail

DRY_RUN=false
PURGE=false
WORKSPACE="${PLAYBOOK_WORKSPACE:-$HOME}"

for arg in "$@"; do
    case "$arg" in
        --dry-run)            DRY_RUN=true ;;
        --purge-downloads)    PURGE=true ;;
        --workspace=*)        WORKSPACE="${arg#*=}" ;;
        --workspace)          ;; # handled below with next arg
        -h|--help)
            cat <<'HELPEOF'
Usage: bash teardown.sh [OPTIONS]

Options:
  (no flags)          Kill running demo processes only
  --dry-run           Show what would happen without doing it
  --purge-downloads   Kill processes AND remove all downloaded assets:
                        - Amazon raw + processed data  (~46 GB)
                        - Base models (TinyLlama) (~2 GB)
                        - Ollama nemotron-mini model   (~3 GB)
                        - HLLM code + dataset + info   (~1 GB)
                        - Project .venv                (~2+ GB)
                      Preserves:
                        - Saved checkpoints ($WORKSPACE/checkpoints/)
                        - This repo's source code
                        - uv and ollama binaries
  --workspace PATH    Root workspace directory (default: $HOME)
                      Use when artifacts live on a different mount, e.g.:
                        bash teardown.sh --purge-downloads --workspace /raid/recsys-playbook
  -h, --help          Show this help

Environment:
  PLAYBOOK_WORKSPACE   Same as --workspace (CLI flag takes precedence)
HELPEOF
            exit 0
            ;;
        *)
            # Handle --workspace as two separate args
            if [[ "${prev_arg:-}" == "--workspace" ]]; then
                WORKSPACE="$arg"
            else
                echo "Unknown option: $arg (run --help for usage)"
                exit 1
            fi
            ;;
    esac
    prev_arg="$arg"
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HLLM_CODE_DIR="$WORKSPACE/hllm-code"
DATA_DIR="$WORKSPACE/data"
MODELS_DIR="$WORKSPACE/models"
CHECKPOINTS_DIR="$WORKSPACE/checkpoints"

echo "Workspace: $WORKSPACE"
echo ""

# ===================================================================
# Phase 1: Kill processes (always runs)
# ===================================================================

killed=0

kill_by_pattern() {
    local label="$1"
    local pattern="$2"
    local pids
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  $label"
        for pid in $pids; do
            if [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ]; then
                continue
            fi
            cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
            cmd=${cmd:0:80}
            echo "    PID $pid: $cmd"
            if ! $DRY_RUN; then
                kill "$pid" 2>/dev/null && killed=$((killed + 1)) || true
            else
                killed=$((killed + 1))
            fi
        done
    fi
}

echo "Stopping demo processes..."
echo ""

kill_by_pattern "Ollama server" "ollama serve"
kill_by_pattern "vLLM server" "vllm.entrypoints"
kill_by_pattern "HLLM training (torchrun)" "torchrun.*run.py"
kill_by_pattern "HLLM training (deepspeed)" "deepspeed.*run.py"
kill_by_pattern "HLLM embedding extraction" "extract_embeddings.py"
kill_by_pattern "FastAPI app (uvicorn)" "uvicorn.*app:app"
kill_by_pattern "W&B agent" "wandb-service"

echo ""
if [ "$killed" -gt 0 ]; then
    if $DRY_RUN; then
        echo "Would kill $killed process(es)."
    else
        echo "Killed $killed process(es)."
        sleep 2
        stragglers=$(pgrep -f "ollama serve|vllm.entrypoints|torchrun.*run.py|uvicorn.*app:app" 2>/dev/null || true)
        if [ -n "$stragglers" ]; then
            echo "Stragglers still running (sending SIGKILL):"
            for pid in $stragglers; do
                cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
                cmd=${cmd:0:80}
                echo "  PID $pid: $cmd"
                kill -9 "$pid" 2>/dev/null || true
            done
        fi
    fi
else
    echo "No demo processes found running."
fi

# ===================================================================
# Phase 2: Remove downloaded assets (only with --purge-downloads)
# ===================================================================

if ! $PURGE; then
    exit 0
fi

echo ""
echo "============================================================"
echo "  --purge-downloads: removing installed assets"
echo "============================================================"

echo ""
echo "Will remove:"
found=0

check_dir() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        local size
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        echo "  $label: $path ($size)"
        return 0
    fi
    return 1
}

check_dir "Amazon raw data"       "$DATA_DIR/raw"                 && found=$((found + 1))
check_dir "Amazon processed data" "$DATA_DIR/processed"           && found=$((found + 1))
check_dir "Base models"           "$MODELS_DIR"                   && found=$((found + 1))
check_dir "HLLM code"             "$HLLM_CODE_DIR"                && found=$((found + 1))
check_dir "HLLM tensorboard logs" "$HLLM_CODE_DIR/log_tensorboard" && found=$((found + 1))
check_dir "HLLM wandb logs"       "$HLLM_CODE_DIR/wandb"          && found=$((found + 1))
check_dir "Project .venv"         "$REPO_DIR/.venv"               && found=$((found + 1))
check_dir "Project wandb logs"    "$REPO_DIR/wandb"               && found=$((found + 1))
check_dir "Project tensorboard"   "$REPO_DIR/log_tensorboard"     && found=$((found + 1))

if ollama list 2>/dev/null | grep -q "nemotron-mini"; then
    echo "  Ollama nemotron-mini model (~2.7 GB)"
    found=$((found + 1))
fi

echo ""
echo "Will preserve:"
echo "  Saved checkpoints: $CHECKPOINTS_DIR/ (trained model weights)"
echo "  Repo source code:  $REPO_DIR"
echo "  uv binary:         $(which uv 2>/dev/null || echo 'not installed')"
echo "  ollama binary:     $(which ollama 2>/dev/null || echo 'not installed')"

if [ "$found" -eq 0 ]; then
    echo ""
    echo "Nothing to remove."
    exit 0
fi

# Require explicit confirmation unless dry-run
if ! $DRY_RUN; then
    echo ""
    read -r -p "Proceed? This will delete all of the above. [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

remove_dir() {
    local label="$1"
    local path="$2"
    if [ -d "$path" ]; then
        if $DRY_RUN; then
            echo "  [dry-run] would remove $path"
        else
            echo "  Removing $path ..."
            rm -rf "$path"
        fi
    fi
}

echo ""

# --- Amazon data ---
remove_dir "Amazon raw data"       "$DATA_DIR/raw"
remove_dir "Amazon processed data" "$DATA_DIR/processed"
if [ -d "$DATA_DIR" ] && [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    remove_dir "Data dir" "$DATA_DIR"
fi

# --- Base models (NOT checkpoints) ---
remove_dir "Base models" "$MODELS_DIR"

# --- HLLM code (preserve $CHECKPOINTS_DIR) ---
remove_dir "HLLM code" "$HLLM_CODE_DIR"

# --- Ollama model ---
if ollama list 2>/dev/null | grep -q "nemotron-mini"; then
    if $DRY_RUN; then
        echo "  [dry-run] would run: ollama rm nemotron-mini"
    else
        echo "  Removing Ollama nemotron-mini model..."
        ollama rm nemotron-mini 2>/dev/null || true
    fi
fi

# --- Project environments and logs ---
remove_dir "Project .venv"       "$REPO_DIR/.venv"
remove_dir "Project wandb logs"  "$REPO_DIR/wandb"
remove_dir "Project tensorboard" "$REPO_DIR/log_tensorboard"

echo ""
if $DRY_RUN; then
    echo "Dry run complete. Run without --dry-run to execute."
else
    echo "Purge complete."
    echo ""
    echo "Preserved:"
    if [ -d "$CHECKPOINTS_DIR" ]; then
        echo "  $CHECKPOINTS_DIR/ ($(du -sh "$CHECKPOINTS_DIR" 2>/dev/null | cut -f1))"
    fi
    echo "  $REPO_DIR (repo source)"
    echo ""
    echo "To fully rebuild: bash $REPO_DIR/assets/setup.sh --workspace $WORKSPACE"
fi
