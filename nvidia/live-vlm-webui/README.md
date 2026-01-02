# Live VLM WebUI

> Real-time Vision Language Model interaction with webcam streaming

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Command Line Options](#command-line-options)
  - [Accept the SSL Certificate](#accept-the-ssl-certificate)
  - [Grant Camera Permissions](#grant-camera-permissions)
  - [Performance Optimization Tips](#performance-optimization-tips)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Live VLM WebUI is a universal web interface for real-time Vision Language Model (VLM) interaction and benchmarking. It enables you to stream your webcam directly to any VLM backend (Ollama, vLLM, SGLang, or cloud APIs) and receive live AI-powered analysis. This tool is perfect for testing VLM models, benchmarking performance across different hardware configurations, and exploring vision AI capabilities.

The interface provides WebRTC-based video streaming, integrated GPU monitoring, customizable prompts, and support for multiple VLM backends. It works seamlessly with the powerful Blackwell GPU in your DGX Spark, enabling real-time vision inference at impressive speeds.

## What you'll accomplish

You'll set up a complete real-time vision AI testing environment on your DGX Spark that allows you to:

- Stream webcam video and get instant VLM analysis through a web browser
- Test and compare different vision language models (Gemma 3, Llama Vision, Qwen VL, etc.)
- Monitor GPU and system performance in real-time while models process video frames
- Customize prompts for various use cases (object detection, scene description, OCR, safety monitoring)
- Access the interface from any device on your network with a web browser

## What to know before starting

- Basic familiarity with Linux command line and terminal operations
- Basic knowledge of Python package installation with pip
- Basic knowledge of REST APIs and how services communicate via HTTP
- Familiarity with web browsers and network access (IP addresses, ports)
- Optional: Knowledge of Vision Language Models and their capabilities (helpful but not required)

## Prerequisites

**Hardware Requirements:**
- Webcam (laptop built-in camera, USB camera, or remote browser with camera)
- At least 10GB available storage space for Python packages and model downloads

**Software Requirements:**
- DGX Spark with DGX OS installed
- Python 3.10 or later (verify with `python3 --version`)
- pip package manager (verify with `pip --version`)
- Network access to download Python packages from PyPI
- A VLM backend running locally (Ollama being easiest) or cloud API access
- Web browser access to `https://<SPARK_IP>:8090`

**VLM Backend Options:**
1. **Ollama** (recommended for beginners) - Easy to install and use
2. **vLLM** - Higher performance for production workloads
3. **SGLang** - Alternative high-performance backend
4. **NIM** - NVIDIA Inference Microservices for optimized performance
5. **Cloud APIs** - NVIDIA API Catalog, OpenAI, or other OpenAI-compatible APIs

## Ancillary files

All source code and documentation can be found at the [Live VLM WebUI GitHub repository](https://github.com/NVIDIA-AI-IOT/live-vlm-webui).

The package will be installed directly via pip, so no additional files are required for basic installation.

## Time & risk

* **Estimated time:** 20-30 minutes (including Ollama installation and model download)
  * 5 minutes to install Live VLM WebUI via pip
  * 10-15 minutes to install Ollama and download a model (varies by model size)
  * 5 minutes to configure and test
* **Risk level:** Low
  * Python packages installed in user space, isolated from system
  * No system-level changes required
  * Port 8090 must be accessible for web interface functionality
  * Self-signed SSL certificate requires browser security exception
* **Rollback:** Uninstall the Python package with `pip uninstall live-vlm-webui`. Ollama can be uninstalled with standard package removal. No persistent changes to DGX Spark configuration.
* **Last Updated:** December 2025
  * First Publication

## Instructions

## Step 1. Install Ollama as VLM Backend

First, install Ollama to serve Vision Language Models. Ollama is one of the easiest options to run/serve models locally on your DGX Spark.

```bash
## Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

## Verify installation
ollama --version
```

Ollama will automatically start as a system service and detect your Blackwell GPU.

Now download a vision language model. We recommend starting with `gemma3:4b` for quick testing:

```bash
## Download a lightweight model (recommended for testing)
ollama pull gemma3:4b

## Alternative models you can try:
## ollama pull llama3.2-vision:11b    # Sometime better quality, slower
## ollama pull qwen2.5-vl:7b          #
```

The model download may take 5-15 minutes depending on your network speed and model size.

Verify Ollama is working:

```bash
## Check if Ollama API is accessible
curl http://localhost:11434/v1/models
```

Expected output should show a JSON response listing your downloaded models.

## Step 2. Install Live VLM WebUI

Install Live VLM WebUI using pip:

```bash
pip install live-vlm-webui
```

The installation will download all required Python dependencies and install the `live-vlm-webui` command.

Now start the server:

```bash
## Launch the web server
live-vlm-webui
```

The server will:
- Auto-generate SSL certificates for HTTPS (required for webcam access)
- Start the WebRTC server on port 8090
- Detect your Blackwell GPU automatically

The server will start and display output like:

```
Starting Live VLM WebUI...
Generating SSL certificates...
GPU detected: NVIDIA GB10 Blackwell

Access the WebUI at:
  Local URL:   https://localhost:8090
  Network URL: https://<YOUR_SPARK_IP>:8090

Press Ctrl+C to stop the server
```

### Command Line Options

Live VLM WebUI supports several command-line options for customization:

```bash
## Specify a different port
live-vlm-webui --port 8091

## Use custom SSL certificates
live-vlm-webui --ssl-cert /path/to/cert.pem --ssl-key /path/to/key.pem

## Change default API endpoint
live-vlm-webui --default-api-base http://localhost:8000/v1

## Run in background (optional)
nohup live-vlm-webui > live-vlm.log 2>&1 &
```

## Step 3. Access the Web Interface

Open your web browser and navigate to:

```
https://<YOUR_SPARK_IP>:8090
```

Replace `<YOUR_SPARK_IP>` with your DGX Spark's IP address. You can find it with:

```bash
hostname -I | awk '{print $1}'
```

**Important:** You must use `https://` (not `http://`) because modern browsers require secure connections for webcam access.

### Accept the SSL Certificate

Since the application uses a self-signed SSL certificate, your browser will show a security warning. This is expected and safe.

**In Chrome/Edge:**
1. Click "**Advanced**" button
2. Click "**Proceed to \<YOUR_SPARK_IP\> (unsafe)**"

**In Firefox:**
1. Click "**Advanced...**"
2. Click "**Accept the Risk and Continue**"

### Grant Camera Permissions

When prompted, allow the website to access your camera. The webcam stream should appear in the interface.

> [!TIP]
> **Remote Access Recommended:** For the best experience, access the web interface from a laptop or PC on the same network. This provides better browser performance and built-in webcam access compared to accessing locally on the DGX Spark.

## Step 4. Configure VLM Settings

The interface auto-detects local VLM backends. Verify the configuration in the **VLM API Configuration** section on the left sidebar:

**API Endpoint:** Should show `http://localhost:11434/v1` (Ollama)

**Model Selection:** Click the dropdown and select your downloaded model (e.g., `gemma3:4b`)

**Optional Settings:**
- **Max Tokens:** Controls response length (default: 512, reduce to 100-200 for faster responses)
- **Frame Processing Interval:** How many frames to skip between analyses (default: 30 frames, increase for slower pace)

### Performance Optimization Tips

For the best performance on DGX Spark Blackwell GPU:

- **Model Selection:** `gemma3:4b` gives 1-2s/frame, `llama3.2-vision:11b` gives slower speed.
- **Frame Interval:** Set to 60 frames (2 seconds at 30 fps) or bigger for comfortable viewing
- **Max Tokens:** Reduce to 100 for faster responses

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Step 5. Start Analyzing Video

Click the green "**Start Camera and Start VLM Analysis**" button.

The interface will:
1. Start streaming your webcam via WebRTC
2. Begin processing frames and sending them to the VLM
3. Display AI analysis results in real-time
4. Show GPU/CPU/RAM metrics at the bottom

You should see:
- **Live video feed** on the right side (with mirror toggle)
- **VLM analysis results** overlaid on video or in the info box
- **Performance metrics** showing latency and frame count
- **GPU monitoring** showing Blackwell GPU utilization and VRAM usage

With the Blackwell GPU in DGX Spark, you should see inference times of **1-2 seconds per frame** for `gemma3:4b` and similar speeds for `llama3.2-vision:11b`.

## Step 6. Customize Prompts

The **Prompt Editor** at the bottom of the left sidebar allows you to customize what the VLM analyzes.

**Quick Prompts** - 8 presets ready to use:
- **Scene Description** - "Describe what you see in this image in one sentence."
- **Object Detection** - "List all objects you can see in this image, separated by commas."
- **Activity Recognition** - "Describe the person's activity and what they are doing."
- **Safety Monitoring** - "Are there any safety hazards visible? Answer with 'ALERT: description' or 'SAFE'."
- **OCR / Text Recognition** - "Read and transcribe any text visible in the image."
- And more...

**Custom Prompts** - Enter your own:

Try this for real-time CSV output (useful for downstream applications):

```
List all objects you can see in this image, separated by commas.
Do not include explanatory text. Output only the comma-separated list.
```

The VLM will immediately start using the new prompt for the next frame analysis. This enables real-time "prompt engineering" where you can iterate and refine prompts while watching live results.

## Step 7. Test Different Models (Optional)

Want to compare models? Download another model and switch:

```bash
## Download another model
ollama pull llama3.2-vision:11b

## The model will appear in the Model dropdown in the web interface
```

In the web interface:
1. Stop VLM analysis (if running)
2. Select the new model from the **Model** dropdown
3. Start VLM analysis again

Compare inference speed and quality between models on your DGX Spark's Blackwell GPU.

## Step 8. Monitor Performance

The bottom section shows real-time system metrics:

- **GPU Usage** - Blackwell GPU utilization percentage
- **VRAM Usage** - GPU memory consumption
- **CPU Usage** - System CPU utilization
- **System RAM** - Memory usage

Use these metrics to:
- Benchmark different models on the same hardware
- Identify performance bottlenecks
- Optimize settings for your use case

## Step 9. Cleanup

When you're done, stop the server with `Ctrl+C` in the terminal where it's running.

To completely remove Live VLM WebUI:

```bash
pip uninstall live-vlm-webui
```

Your Ollama installation and downloaded models remain available for future use.

To remove Ollama as well (optional):

```bash
## Uninstall Ollama
sudo systemctl stop ollama
sudo rm /usr/local/bin/ollama
sudo rm -rf /usr/share/ollama

## Remove Ollama models (optional)
rm -rf ~/.ollama
```

## Step 10. Next Steps

Now that you have Live VLM WebUI running, explore these use cases:

**Model Benchmarking:**
- Test multiple models (Gemma 3, Llama Vision, Qwen VL) on your DGX Spark
- Compare inference latency, accuracy, and GPU utilization
- Evaluate structured output capabilities (JSON, CSV)

**Application Prototyping:**
- Use the web interface as reference for building your own VLM applications
- Integrate with ROS 2 for robotics vision
- Connect to RTSP IP cameras for security monitoring (Beta feature)

**Cloud API Integration:**
- Switch from local Ollama to cloud APIs (NVIDIA API Catalog, OpenAI)
- Compare edge vs. cloud inference performance and costs
- Test hybrid deployments

To use NVIDIA API Catalog or other cloud APIs:

1. In the **VLM API Configuration** section, change the **API Base URL** to:
   - NVIDIA API Catalog: `https://integrate.api.nvidia.com/v1`
   - OpenAI: `https://api.openai.com/v1`
   - Other: Your custom endpoint

2. Enter your **API Key** in the field that appears

3. Select your model from the dropdown (list is fetched from the API)

**Advanced Configuration:**
- Use vLLM, SGLang, or NIM backends for higher throughput
- Set up NIM for optimized NVIDIA-specific performance
- Customize the Python backend for your specific use case

For more advanced usage, see the [full documentation](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/tree/main/docs) on GitHub.

For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) and the [Live VLM WebUI Troubleshooting Guide](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/docs/troubleshooting.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| pip install shows "error: externally-managed-environment" | Python 3.12+ prevents system-wide pip installs | Use virtual environment: `python3 -m venv live-vlm-env && source live-vlm-env/bin/activate && pip install live-vlm-webui` |
| Browser shows "Your connection is not private" warning | Application uses self-signed SSL certificate | Click "Advanced" → "Proceed to \<IP\> (unsafe)" - this is safe and expected behavior |
| Camera not accessible or "Permission Denied" | Browser requires HTTPS for webcam access | Ensure you're using `https://` (not `http://`). Accept self-signed certificate warning and grant camera permissions when prompted |
| "Failed to connect to VLM" or "Connection refused" | Ollama or VLM backend not running | Verify Ollama is running with `curl http://localhost:11434/v1/models`. If not running, start with `sudo systemctl start ollama` |
| VLM responses are very slow (>5 seconds per frame) | Model too large for available VRAM or incorrect configuration | Try a smaller model (`gemma3:4b` instead of larger models). Increase Frame Processing Interval to 60+ frames. Reduce Max Tokens to 100-200 |
| GPU stats show "N/A" for all metrics | NVML not available or GPU driver issues | Verify GPU access with `nvidia-smi`. Ensure NVIDIA drivers are properly installed |
| "No models available" in model dropdown | API endpoint incorrect or models not downloaded | Verify API endpoint is `http://localhost:11434/v1` for Ollama. Download models with `ollama pull gemma3:4b` |
| Server fails to start with "port already in use" | Port 8090 already occupied by another service | Stop the conflicting service or use `--port` flag to specify a different port: `live-vlm-webui --port 8091` |
| Cannot access from remote browser on network | Firewall blocking port 8090 or wrong IP address | Verify firewall allows port 8090: `sudo ufw allow 8090`. Use correct IP from `hostname -I` command |
| Video stream is laggy or frozen | Network issues or browser performance | Use Chrome or Edge browser. Access from a separate PC on the network rather than locally. Check network bandwidth |
| Analysis results in unexpected language | Model supports multilingual and detected language in prompt | Explicitly specify output language in prompt: "Answer in English: describe what you see" |
| pip install fails with dependency errors | Conflicting Python package versions | Try installing with `--user` flag: `pip install --user live-vlm-webui` |
| Command `live-vlm-webui` not found after install | Binary path not in PATH | Add `~/.local/bin` to PATH: `export PATH="$HOME/.local/bin:$PATH"` then run `source ~/.bashrc` |
| Camera works but no VLM analysis results appear, browser shows InvalidStateError | Accessing via SSH port forwarding from remote machine | WebRTC requires direct network connectivity and doesn't work through SSH tunnels (SSH only forwards TCP, WebRTC needs UDP). **Solution 1**: Access the web UI directly from a browser on the same network as the server. **Solution 2**: Use the server machine's browser directly. **Solution 3**: Use X11 forwarding (`ssh -X`) to display the browser remotely |
