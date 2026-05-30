
# SGLang Setup on DGX Station

Deploy an SGLang inference server on DGX Station with validated configuration.

## Steps

1. **Find the GB300 GPU index.** Run:
   ```bash
   nvidia-smi --query-gpu=index,name --format=csv,noheader
   ```
   Identify the device index for the GB300 (typically device 1). Use this index for `--gpus` below. Do NOT use `--gpus all` — mixed coherency will cause CUDA failures.

2. **Ask the user which model to serve.** If they don't have a preference, suggest:
   - `Qwen/Qwen3-8B` — small, fast, good for testing
   - `Qwen/Qwen3-32B` — medium, good balance
   - `meta-llama/Llama-3.1-70B-Instruct` — large general-purpose

3. **Check if the user has an HF_TOKEN.** Pass inline with `-e HF_TOKEN="..."`.

4. **Deploy the container.** Use this validated configuration:

   ```bash
   docker pull lmsysorg/sglang:latest-cu130

   docker run -d \
     --name sglang-server \
     --gpus '"device=<GB300_INDEX>"' \
     --ipc host \
     --ulimit memlock=-1 \
     --ulimit stack=67108864 \
     -p 30000:30000 \
     -e HF_TOKEN="<TOKEN>" \
     -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
     lmsysorg/sglang:latest-cu130 \
     sglang serve --model-path "<MODEL>" \
       --host 0.0.0.0 \
       --port 30000 \
       --context-length 32768 \
       --mem-fraction-static 0.85
   ```

   **Container version:** Use `lmsysorg/sglang:latest-cu130`. The `cu130` tag is required for Blackwell SM103 support.

   **First launch** downloads the model and compiles kernels. This takes extra time — subsequent starts are faster.

5. **Wait for the server to be ready.** Monitor logs:
   ```bash
   docker logs -f sglang-server
   ```

6. **Test the server:**
   ```bash
   curl http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "<MODEL>",
       "messages": [{"role": "user", "content": "Hello"}],
       "max_tokens": 64
     }'
   ```

7. **Report the result** to the user, including:
   - Model loaded and serving on port 30000
   - How to stop: `docker stop sglang-server && docker rm sglang-server`

## Key features

- **RadixAttention** — automatic KV cache reuse across requests sharing prefixes. On by default, no flag needed. Verify with: `docker logs sglang-server 2>&1 | grep "cached-token" | tail -5`
- **Structured JSON output** — use `response_format.json_schema` in API requests for guaranteed valid JSON.
- **Chunked prefill** — add `--chunked-prefill-size 8192` to break long prefills into chunks, reducing time-to-first-token.

## Tuning parameters

| Parameter | Default | Agent workloads | Throughput workloads |
|-----------|---------|-----------------|---------------------|
| `--context-length` | 32768 | 32768-65536 | 8192-16384 |
| `--mem-fraction-static` | 0.85 | 0.80-0.85 | 0.85-0.88 |
| `--chunked-prefill-size` | off | 4096-8192 | 8192 |
| `--enable-metrics` | off | Optional | Recommended |

## Structured output example

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL>",
    "messages": [{"role": "user", "content": "List three programming languages."}],
    "max_tokens": 512,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "languages",
        "schema": {
          "type": "object",
          "properties": {
            "languages": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "primary_use": {"type": "string"}
                },
                "required": ["name", "primary_use"]
              }
            }
          },
          "required": ["languages"]
        }
      }
    }
  }'
```
