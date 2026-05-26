# Patches for Isaac GR00T (N1.6 playbook)

Apply from the root of a **clean clone** of [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) on branch **`n1.6-release`** (see playbook Instructions — clone/checkout). Upstream already sets backend fallback order to **torchcodec → decord → pyav → ffmpeg**; this patch adds the missing **`get_frames_by_indices`** implementation for **`pyav`**, which avoids CPU-bound **ffmpeg** subprocess fallback during LIBERO training.

## `001-pyav-get-frames-by-indices.patch`

**When:** `torchcodec` is missing or fails to import, the resolver falls back to **`pyav`**, and training or evaluation raises:

`NotImplementedError` from `get_frames_by_indices` (PyAV was listed in the fallback chain but had no index-based reader).

**Apply:**

```bash
cd Isaac-GR00T
git checkout n1.6-release
git apply /absolute/path/to/dgx-station-playbooks/nvidia/station-gr00t/assets/patches/001-pyav-get-frames-by-indices.patch
```

Or copy this playbook’s `nvidia/station-gr00t/assets/patches/` directory into your clone and run `git apply assets/patches/001-pyav-get-frames-by-indices.patch` from the Isaac-GR00T repo root.

Re-applying the same patch fails until you revert `gr00t/utils/video_utils.py` (for example `git checkout -- gr00t/utils/video_utils.py`).

**After patching:** ensure PyAV is installed in the project venv (`uv pip install av`).
