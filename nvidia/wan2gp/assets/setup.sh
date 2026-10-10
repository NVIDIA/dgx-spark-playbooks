#!/usr/bin/env bash
# =============================================================================
#  Wan2GP (WanGP) — DGX Spark setup
# -----------------------------------------------------------------------------
#  Installs deepbeepmeep's Wan2GP AI video generator natively on a DGX Spark
#  (GB10 Grace-Blackwell, aarch64, CUDA 13), including the InfiniteTalk /
#  MultiTalk audio-driven talking-head models.
#
#  Design notes for the Spark / aarch64 / Blackwell:
#    * PyTorch is installed from the official cu130 wheel index (sm_121 build),
#      the same source NVIDIA's ComfyUI playbook uses. This is the piece that
#      normally breaks on ARM+Blackwell, so we reuse the known-good wheels.
#    * A DEDICATED venv (~/wan2gp-env) is created so an existing ComfyUI
#      install (~/comfyui-env) is never disturbed.
#    * SageAttention / FlashAttention / xformers are skipped — they have no
#      ARM wheels. Wan2GP falls back to PyTorch's built-in sdpa attention.
#    * A few of Wan2GP's requirements pin fast-moving nightly builds
#      (torchcodec, onnxruntime-gpu) that rotate off the index, or have no ARM
#      wheel at all (decord); these are handled explicitly below.
#    * transformers is pinned to 4.54.0 (Wan2GP's target); newer 5.x ships a
#      model that collides with Wan2GP's bundled OmniVoice TTS.
#
#  Tested on: DGX Spark (GB10), DGX OS, Python 3.12, PyTorch 2.13.0+cu130.
# =============================================================================
set -euo pipefail

CU130="https://download.pytorch.org/whl/cu130"
ENV_DIR="${WAN2GP_ENV_DIR:-$HOME/wan2gp-env}"
REPO_DIR="${WAN2GP_REPO_DIR:-$HOME/Wan2GP}"

echo "=== [1/7] System packages ==="
# python3.12-dev + build-essential give Triton the Python.h headers it needs to
# JIT-compile CUDA kernels (otherwise it silently rolls back to CPU).
sudo apt-get update -y
sudo apt-get install -y git ffmpeg build-essential python3.12-dev

echo "=== [2/7] Create dedicated venv at $ENV_DIR ==="
[ -d "$ENV_DIR" ] || python3.12 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "=== [3/7] Install Blackwell PyTorch from the cu130 index ==="
# Validated versions for the Spark; bump together if you retest a newer set.
pip install --index-url "$CU130" "torch==2.13.0" "torchvision==0.28.0" torchaudio

echo "=== [4/7] Clone / update Wan2GP ==="
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only || true
else
  git clone https://github.com/deepbeepmeep/Wan2GP.git "$REPO_DIR"
fi

echo "=== [5/7] Install Wan2GP dependencies (torch pinned; ARM-incompatible libs skipped) ==="
cat > /tmp/wan2gp-constraints.txt <<EOF
torch==2.13.0+cu130
torchvision==0.28.0+cu130
transformers==4.54.0
EOF
# Strip torch* (incl. torchcodec), the pinned onnxruntime nightlies, decord
# (no ARM wheel), rembg's [gpu] extra, and compile-only attention libraries.
# These are installed with working substitutes in step [6].
# Boundary is "any non-package-name character or end of line" so it also catches
# environment markers (decord; ...) and extras (rembg[gpu]).
grep -viE '^[[:space:]]*(torch|torchvision|torchaudio|torchcodec|onnxruntime-gpu|onnxruntime|decord|rembg|sageattention|flash[_-]?attn|flash-attention|xformers)([^A-Za-z0-9._-]|$)' \
  "$REPO_DIR/requirements.txt" > /tmp/wan2gp-reqs.txt || true

# Fast path: a fully-resolved install. If it fails, retry line-by-line so every
# remaining problem package surfaces at once instead of one-per-run.
if ! pip install -r /tmp/wan2gp-reqs.txt -c /tmp/wan2gp-constraints.txt; then
  echo ">>> Batch install failed; retrying line-by-line to surface all problem packages..."
  : > /tmp/wan2gp-failed.txt
  while IFS= read -r line; do
    [ -z "${line//[[:space:]]/}" ] && continue
    case "$line" in \#*) continue;; esac
    pip install "$line" -c /tmp/wan2gp-constraints.txt >/dev/null 2>&1 \
      || echo "$line" >> /tmp/wan2gp-failed.txt
  done < /tmp/wan2gp-reqs.txt
  [ -s /tmp/wan2gp-failed.txt ] && { echo ">>> Could not install:"; cat /tmp/wan2gp-failed.txt; }
fi

echo "=== [6/7] Install ARM-friendly substitutes ==="
# torchcodec: pinned version has no ARM/py3.12 wheel; take the cu130-matched one.
pip install --index-url "$CU130" torchcodec || pip install torchcodec || \
  echo "WARN: torchcodec not installed (only needed for video-INPUT decoding)."
# onnxruntime: GPU build has no ARM wheel; CPU build serves the ONNX helpers.
pip install onnxruntime-gpu || pip install onnxruntime || \
  echo "WARN: onnxruntime not installed (background-removal / audio-separation helpers)."
# rembg: install the CPU build (the [gpu] extra requires ARM-less onnxruntime-gpu).
pip install "rembg==2.0.65" || pip install rembg || \
  echo "WARN: rembg not installed (wgp.py imports it at startup)."
# audio-separator: needed by InfiniteTalk/MultiTalk vocal extraction.
pip install "audio-separator==0.36.1" || pip install --no-deps "audio-separator==0.36.1" || \
  echo "WARN: audio-separator not installed (InfiniteTalk/MultiTalk preprocessing)."

echo "=== [7/7] Sanity check: PyTorch sees the Blackwell GPU ==="
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device         :", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("compute cap    :", f"sm_{cap[0]}{cap[1]}")
PY

echo
echo "============================================================"
echo " Wan2GP install complete."
echo " Launch with:  bash \"$(dirname "$0")/launch.sh\""
echo "============================================================"
