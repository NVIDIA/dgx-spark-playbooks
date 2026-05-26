FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /opt/ComfyUI

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git .

# Install ComfyUI dependencies WITHOUT overwriting NGC-optimized PyTorch.
# Strip torch/torchvision/torchaudio lines from ComfyUI requirements (pip must not replace NGC wheels).
RUN sed -i '/^torch\b/d; /^torchvision\b/d; /^torchaudio\b/d; /^nvidia/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Provide an import-only torchaudio stub. NGC PyTorch 26.02 exports custom
# ABI symbols (e.g. torch_dtype_float4_e2m1fn_x2) that PyPI's torchaudio
# wheel does not export, so `pip install torchaudio` produces an
# unloadable libtorchaudio.abi3.so. ComfyUI's comfy/sd.py unconditionally
# imports comfy.ldm.lightricks.vae.audio_vae which imports torchaudio,
# so the import has to succeed even though we don't use audio VAE.
# Lightricks audio VAE is not exercised by any workflow in this playbook.
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p "$SITE/torchaudio" && \
    printf '%s\n' \
        '# Audio VAE stub for NGC PyTorch images. See assets/Dockerfile for context.' \
        '__version__ = "0.0.0+ngc-stub"' \
        '' \
        'class _Unavailable:' \
        '    def __getattr__(self, name):' \
        '        raise RuntimeError(' \
        '            "torchaudio is stubbed in this NGC image; audio workflows are not supported."' \
        '        )' \
        '' \
        'functional = _Unavailable()' \
        'transforms = _Unavailable()' \
        > "$SITE/torchaudio/__init__.py" && \
    python -c "import torchaudio; assert torchaudio.__version__.endswith('ngc-stub')"

# Install custom nodes
RUN cd custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git && \
    git clone https://github.com/lum3on/comfyui_HiDream-Sampler.git && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# Install custom node dependencies.
#
# Two pre-processing fixes before the install loop:
#  1. onnxruntime-gpu has no aarch64 wheel on PyPI; on Grace (arm64)
#     the install fails and pip skips the package. controlnet_aux
#     preprocessors then crash at runtime with "No module named
#     onnxruntime". Swap it for the plain `onnxruntime` (CPU) wheel,
#     which IS published for aarch64 — preprocessors then run on CPU.
#  2. Some custom nodes pin packages that conflict with the NGC torch
#     constraint set; we drop torch/torchvision/torchaudio lines in
#     each custom-node requirements.txt for the same reason as the
#     ComfyUI requirements step above.
RUN for req in custom_nodes/*/requirements.txt; do \
      [ -f "$req" ] || continue; \
      sed -i 's/^onnxruntime-gpu\b/onnxruntime/' "$req"; \
      sed -i '/^torch\b/d; /^torchvision\b/d; /^torchaudio\b/d' "$req"; \
    done && \
    for req in custom_nodes/*/requirements.txt; do \
      [ -f "$req" ] && pip install --no-cache-dir -r "$req" || true; \
    done

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
