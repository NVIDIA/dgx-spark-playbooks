#!/usr/bin/env bash
# download-models.sh — Download all model weights for the ComfyUI playbook.
#
# Usage:
#   export HF_TOKEN="your_huggingface_token"
#   bash assets/scripts/download-models.sh [TIER]
#
# TIER (optional):
#   1  — FLUX.1 dev + Wan 2.1 T2V 14B (Getting Started, ~70 GB)
#   2  — Tier 1 + HiDream-I1 + Wan 2.1 I2V + Cosmos-Predict2 (~180 GB)
#   3  — All models including HunyuanVideo + ControlNet (~230 GB)
#
# Default: downloads all tiers.
#
# Models are downloaded to ./models/ subdirectories matching ComfyUI's layout:
#   models/diffusion_models/  models/text_encoders/  models/vae/  models/clip_vision/
#
# Requires: huggingface_hub CLI (`pip install huggingface_hub` → `hf` command).

set -euo pipefail

TIER="${1:-3}"
MODELS_DIR="./models"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set. Export your HuggingFace token first."
  echo "  export HF_TOKEN=\"your_huggingface_token\""
  exit 1
fi

# Auto-prepend ~/.local/bin so the user-pip install of huggingface-hub
# (pip3 install --break-system-packages huggingface-hub) is found even
# when the shell hasn't sourced ~/.bashrc — the binary lands at
# ~/.local/bin/hf which is not on the default non-interactive PATH.
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' CLI not found. Install Hugging Face Hub:"
  echo "  pip3 install --break-system-packages huggingface-hub"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
fi

# Pass HF_TOKEN to hf via env, not --token. hf and huggingface_hub both
# honor HF_TOKEN, and putting the token on the command line leaks it
# into `ps`, syslog, and shell history.
export HF_TOKEN

# Optional 4th argument: when two different repos publish a file with
# the same basename (e.g. ae.safetensors from FLUX vs HiDream), the
# caller can ask hf_download to save under a custom local name. When
# omitted, behavior is identical to the original basename-only copy.
hf_download() {
  local repo="$1" file="$2" dest_dir="$3" local_name="${4:-}"
  local basename staging
  basename=$(basename "$file")
  local dest_name="${local_name:-$basename}"
  if [ -f "$dest_dir/$dest_name" ]; then
    echo "  Already exists: $dest_name"
    return
  fi
  echo "  Downloading: $repo/$file -> $dest_dir/$dest_name"
  mkdir -p "$dest_dir"
  staging=$(mktemp -d)
  cleanup() { rm -rf "$staging"; }
  trap cleanup RETURN

  HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-0}" \
    hf download "$repo" "$file" \
    --local-dir "$staging"

  local src="$staging/$file"
  if [ ! -f "$src" ]; then
    echo "ERROR: Download finished but expected file is missing: $src" >&2
    echo "       Repo: $repo  File: $file" >&2
    exit 1
  fi
  cp "$src" "$dest_dir/$dest_name"
  echo "  Done: $dest_name"
}

echo "Downloading models (Tier $TIER)..."
echo "Models directory: $MODELS_DIR"
echo ""

# ─── Tier 1: Getting Started ────────────────────────────────────────────────

echo "═══ Tier 1: Getting Started ═══"

echo "Downloading FLUX.1 [dev]..."
hf_download "Comfy-Org/flux1-dev" "flux1-dev.safetensors" "$MODELS_DIR/diffusion_models"

echo "Downloading FLUX text encoders..."
hf_download "comfyanonymous/flux_text_encoders" "clip_l.safetensors" "$MODELS_DIR/text_encoders"
hf_download "comfyanonymous/flux_text_encoders" "t5xxl_fp16.safetensors" "$MODELS_DIR/text_encoders"

echo "Downloading FLUX VAE..."
hf_download "black-forest-labs/FLUX.1-dev" "ae.safetensors" "$MODELS_DIR/vae"

echo "Downloading Wan 2.1 T2V 14B..."
hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors" "$MODELS_DIR/diffusion_models"
hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"
hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/vae/wan_2.1_vae.safetensors" "$MODELS_DIR/vae"

echo ""
echo "Tier 1 verification (expect several large .safetensors files):"
ls -la "$MODELS_DIR/diffusion_models" || true
ls -la "$MODELS_DIR/text_encoders" | head -20 || true

if [ "$TIER" -lt 2 ]; then
  echo ""
  echo "Tier 1 download complete."
  exit 0
fi

# ─── Tier 2: Intermediate ───────────────────────────────────────────────────

echo ""
echo "═══ Tier 2: Intermediate Pipelines ═══"

echo "Downloading HiDream-I1 Full..."
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/diffusion_models/hidream_i1_full_fp16.safetensors" "$MODELS_DIR/diffusion_models"
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/text_encoders/clip_l_hidream.safetensors" "$MODELS_DIR/text_encoders"
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/text_encoders/clip_g_hidream.safetensors" "$MODELS_DIR/text_encoders"
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors" "$MODELS_DIR/text_encoders"
# HiDream's VAE has the same basename (ae.safetensors) as FLUX's VAE.
# Save under a distinct local name so the FLUX VAE downloaded in Tier 1
# isn't shadowed and the HiDream workflow can refer to its own VAE.
hf_download "Comfy-Org/HiDream-I1_ComfyUI" "split_files/vae/ae.safetensors" "$MODELS_DIR/vae" "ae_hidream.safetensors"

echo "Downloading Wan 2.1 I2V 14B 720P..."
hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" "$MODELS_DIR/diffusion_models"
hf_download "Comfy-Org/Wan_2.1_ComfyUI_repackaged" "split_files/clip_vision/clip_vision_h.safetensors" "$MODELS_DIR/clip_vision"

echo "Downloading Cosmos-Predict2 14B Video2World..."
hf_download "Comfy-Org/Cosmos_Predict2_repackaged" "cosmos_predict2_14B_video2world_720p_16fps.safetensors" "$MODELS_DIR/diffusion_models"
hf_download "comfyanonymous/cosmos_1.0_text_encoder_and_VAE_ComfyUI" "text_encoders/oldt5_xxl_fp8_e4m3fn_scaled.safetensors" "$MODELS_DIR/text_encoders"

if [ "$TIER" -lt 3 ]; then
  echo ""
  echo "Tier 2 download complete."
  exit 0
fi

# ─── Tier 3: Advanced ───────────────────────────────────────────────────────

echo ""
echo "═══ Tier 3: Advanced Techniques ═══"

echo "Downloading HunyuanVideo..."
hf_download "Comfy-Org/HunyuanVideo_repackaged" "split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors" "$MODELS_DIR/diffusion_models"
# HunyuanVideo's CLIP-L is a different blob than FLUX's clip_l.safetensors.
# Save it under a distinct local name to avoid shadowing the FLUX one.
hf_download "Comfy-Org/HunyuanVideo_repackaged" "split_files/text_encoders/clip_l.safetensors" "$MODELS_DIR/text_encoders" "clip_l_hunyuan.safetensors"
hf_download "Comfy-Org/HunyuanVideo_repackaged" "split_files/text_encoders/llava_llama3_fp8_scaled.safetensors" "$MODELS_DIR/text_encoders"
hf_download "Comfy-Org/HunyuanVideo_repackaged" "split_files/vae/hunyuan_video_vae_bf16.safetensors" "$MODELS_DIR/vae"

echo "Downloading FLUX ControlNet (Canny Dev)..."
hf_download "Comfy-Org/flux1-dev" "split_files/diffusion_models/flux1-canny-dev.safetensors" "$MODELS_DIR/diffusion_models"

echo ""
echo "All models downloaded successfully."
echo "Total disk usage:"
du -sh "$MODELS_DIR"
echo ""
echo "Diffusion models present:"
ls -la "$MODELS_DIR/diffusion_models" || true
