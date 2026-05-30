
# vLLM Setup on DGX Station

Deploy a vLLM inference server on DGX Station with validated configuration.

## Steps

1. **Find the GB300 GPU index.** Run:
   ```bash
   nvidia-smi --query-gpu=index,name --format=csv,noheader
   ```
   Identify the device index for the GB300 (typically device 1). Use this index for `--gpus` below. Do NOT use `--gpus all` — mixed coherency will cause CUDA failures.

2. **Ask the user which model to serve.** If they don't have a preference, suggest:
   - `nvidia/Qwen3-235B-A22B-NVFP4` — large MoE model, fits in 279 GB HBM
   - `meta-llama/Llama-3.1-70B-Instruct` — solid general-purpose model
   - `Qwen/Qwen3-8B` — small model for testing

3. **Check if the user has an HF_TOKEN.** Many models require HuggingFace authentication. The token must be passed inline with `-e HF_TOKEN="..."` — do not rely on shell export in background Docker tasks.

4. **Deploy the container.** Use this validated configuration:

   ```bash
   docker pull nvcr.io/nvidia/vllm:26.01-py3

   docker run -d \
     --name vllm-server \
     --gpus '"device=<GB300_INDEX>"' \
     --ipc host \
     --ulimit memlock=-1 \
     --ulimit stack=67108864 \
     -p 8000:8000 \
     -e HF_TOKEN="<TOKEN>" \
     -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
     nvcr.io/nvidia/vllm:26.01-py3 \
     vllm serve "<MODEL>" \
       --max-model-len 32768 \
       --gpu-memory-utilization 0.9
   ```

   **Container version:** Use `nvcr.io/nvidia/vllm:26.01-py3`. Do NOT use 25.10 — it has a FlashInfer buffer overflow on DGX Station.

5. **Wait for the server to be ready.** Monitor logs:
   ```bash
   docker logs -f vllm-server
   ```
   Wait for the line indicating the server is listening on port 8000.

6. **Test the server:**
   ```bash
   curl http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "<MODEL>",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 64
     }'
   ```

7. **Report the result** to the user, including:
   - Model loaded and serving on port 8000
   - GPU memory utilization
   - How to stop: `docker stop vllm-server && docker rm vllm-server`

## Tuning parameters

Adjust these based on the user's workload:

| Parameter | Default | Agent workloads | Throughput workloads |
|-----------|---------|-----------------|---------------------|
| `--max-model-len` | 32768 | 32768-65536 | 8192-16384 |
| `--gpu-memory-utilization` | 0.9 | 0.85-0.90 | 0.90-0.92 |
| `--enable-prefix-caching` | off | Enable (multi-turn reuse) | Enable |
| `--max-num-seqs` | default | 4-16 (lower latency) | 32+ (higher throughput) |
