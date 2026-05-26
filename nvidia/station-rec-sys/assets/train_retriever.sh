#!/usr/bin/env bash
# train_retriever_1b.sh — Train HLLM retriever with LoRA on TinyLlama-1.1B.
#
# Uses the known-good TinyLlama recipe that produced R@10 = 0.0708 on the
# Amazon Dresses test set (see docs/4-19-metrics-analysis.md run 7sb4tigy).
# Wall-clock: ~20 min per epoch on a single GB300 at ~13.75 s/step (bs=512).
#
# USAGE:
#   bash assets/train_retriever.sh                         # Default: train from scratch (any existing ckpt is ignored)
#   bash assets/train_retriever.sh --resume                # Resume from latest checkpoint in $CHECKPOINT_DIR
#   CUDA_VISIBLE_DEVICES=1 bash assets/train_retriever.sh  # Use GPU 1
#   PLAYBOOK_WORKSPACE=/raid bash assets/train_retriever.sh
#
# RECIPE NOTES:
#   - Model:             TinyLlama-1.1B
#   - train_batch_size:  512 — peak GPU mem ~154 GB / 284 GB on GB300
#   - grad_accum_steps:  1 — no accumulation needed
#   - num_negatives:     4096 — paper-scale NCE signal
#   - MAX_ITEM_LIST_LEN: 20 — longer user histories
#   - learning_rate:     2e-4 — larger LR OK at this scale
#   - gradient_ckpt:     True — cheap safety on a 1B LoRA run
#   - epochs:            1 (~20 min on GB300; set PLAYBOOK_EPOCHS=3-5 for production-quality embeddings)

set -euo pipefail

# ---- CLI flags ----
AUTO_RESUME=False
for arg in "$@"; do
    case "$arg" in
        --resume)
            AUTO_RESUME=True
            ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

# ---- Paths ----
WORKSPACE="${PLAYBOOK_WORKSPACE:-$HOME}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HLLM_CODE_DIR="$WORKSPACE/hllm-code"
DATA_DIR="$WORKSPACE/data"
MODELS_DIR="$WORKSPACE/models"
CHECKPOINTS_DIR="$WORKSPACE/checkpoints"
CHECKPOINT_DIR="${PLAYBOOK_CHECKPOINT_DIR:-$CHECKPOINTS_DIR/dresses_lora_r16}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EPOCHS="${PLAYBOOK_EPOCHS:-1}"
NUM_NEGATIVES="${PLAYBOOK_NUM_NEGATIVES:-4096}"
MAX_STEPS="${PLAYBOOK_MAX_STEPS:-0}"
SAVE_STEPS="${PLAYBOOK_SAVE_STEPS:-200}"
TORCH_COMPILE="${PLAYBOOK_TORCH_COMPILE:-True}"

# Activate uv venv so torchrun uses the correct Python
source "$REPO_DIR/.venv/bin/activate"

# ---- Banner ----
echo "============================================================"
echo "  HLLM Retriever Training (LoRA, TinyLlama-1.1B)"
echo "============================================================"
echo ""
echo "  Model:        TinyLlama-1.1B + LoRA r16"
echo "  Dataset:      Amazon Dresses (293K interactions)"
echo "  GPU:          $GPU_ID"
echo "  Checkpoints:  $CHECKPOINT_DIR"
echo "  Data dir:     $DATA_DIR"
echo "  Epochs:       $EPOCHS"
echo "  Negatives:    $NUM_NEGATIVES"
echo "  Max steps:    $MAX_STEPS (0 = full epoch schedule)"
echo "  Compile:      $TORCH_COMPILE"
if [ "$AUTO_RESUME" = "True" ]; then
    echo "  Resume:       on (--resume: latest checkpoint in $CHECKPOINT_DIR will be loaded)"
else
    echo "  Resume:       off (training from scratch; pass --resume to continue from latest checkpoint)"
fi
echo ""

# Check W&B auth via netrc (what the Python client reads at training time).
# Avoids the wandb CLI, which isn't on PATH unless the venv is active and
# whose `wandb status` output doesn't expose a stable "logged in" string.
if [ -f "$HOME/.netrc" ] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "  W&B:          Enabled (netrc auth) — runs land at https://wandb.ai"
    echo "                Project: enterprise-recsys"
    echo "                Run URL appears in the training log after W&B initializes."
else
    echo "  W&B:          Not logged in (run 'uv run wandb login' to enable)"
fi

echo ""
echo "  Estimated wall time: ~20 min per epoch (~13.75 s/step on GB300 at bs=512)"
echo "  Monitor GPU: open another terminal and run 'watch nvidia-smi'"
echo ""
echo "============================================================"
echo ""

# ---- Launch training ----
cd "$HLLM_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONUNBUFFERED=1 WANDB_API_KEY=${WANDB_TOKEN:-} \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --master_port=12350 --nproc_per_node=1 --nnodes=1 \
  run.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --loss nce \
  \
  --epochs "$EPOCHS" \
  --train_batch_size 512 \
  --gradient_accumulation_steps 1 \
  --save_steps "$SAVE_STEPS" \
  --num_negatives "$NUM_NEGATIVES" \
  --max_steps "$MAX_STEPS" \
  --MAX_TEXT_LENGTH 64 \
  --MAX_ITEM_LIST_LENGTH 20 \
  --gradient_checkpointing True \
  --num_workers 32 \
  --fast_eval_interval 500 \
  \
  --torch_compile "$TORCH_COMPILE" \
  --torch_compile_mode default \
  \
  --use_fused_adam True \
  --optim_args.learning_rate 2e-4 \
  --optim_args.weight_decay 0.05 \
  --scheduler_args.warmup 0.05 \
  \
  --lora_r 16 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules '["q_proj","k_proj","v_proj","o_proj"]' \
  \
  --dataset amazon_dresses \
  --data_path "$HLLM_CODE_DIR/dataset/" \
  --item_pretrain_dir "$MODELS_DIR/TinyLlama-1.1B" \
  --user_pretrain_dir "$MODELS_DIR/TinyLlama-1.1B" \
  --text_path "$HLLM_CODE_DIR/information" \
  --text_keys '["title","description"]' \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  \
  --log_wandb True \
  --wandb_project enterprise-recsys \
  --wandb_log_interval 5 \
  --eval_step 1 \
  --stopping_step 2 \
  --auto_resume "$AUTO_RESUME"
