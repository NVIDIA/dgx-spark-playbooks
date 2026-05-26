# vLLM Service

This service provides advanced GPU-accelerated LLM inference using vLLM with FP8 quantization, offering higher throughput than Ollama for production workloads.

## Overview

vLLM is an optional service that complements Ollama by providing:
- Higher throughput for concurrent requests
- Advanced quantization (FP8)
- PagedAttention for efficient memory usage
- OpenAI-compatible API

## Quick Start

### Using the Complete Stack

The easiest way to run vLLM is with the complete stack:

```bash
# From project root
./start.sh --complete
```

This starts vLLM along with all other optional services.

### Manual Docker Compose

```bash
# From project root
docker compose -f deploy/compose/docker-compose.complete.yml up -d vllm
```

### Testing the Deployment

```bash
# Check health
curl http://localhost:8001/v1/models

# Test chat completion
curl -X POST "http://localhost:8001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "max_tokens": 100
  }'
```

## Default Configuration

- **Model**: `meta-llama/Llama-3.2-3B-Instruct`
- **Quantization**: FP8 (optimized for compute efficiency)
- **Port**: 8001
- **API**: OpenAI-compatible endpoints

## Configuration Options

Environment variables configured in `docker-compose.complete.yml`:

- `VLLM_MODEL`: Model to load (default: `meta-llama/Llama-3.2-3B-Instruct`)
- `VLLM_TENSOR_PARALLEL_SIZE`: Number of GPUs to use (default: 1)
- `VLLM_MAX_MODEL_LEN`: Maximum sequence length (default: 4096)
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory usage (default: 0.9)
- `VLLM_QUANTIZATION`: Quantization method (default: fp8)
- `VLLM_KV_CACHE_DTYPE`: KV cache data type (default: fp8)

## Frontend Integration

The txt2kg frontend automatically detects and uses vLLM when available:

1. Triple extraction: `/api/vllm` endpoint
2. RAG queries: Automatically uses vLLM if configured
3. Model selection: Choose vLLM models in the UI

## Using Different Models

To use a different model, edit the `VLLM_MODEL` environment variable in `docker-compose.complete.yml`:

```yaml
environment:
  - VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

Then restart the service:

```bash
docker compose -f deploy/compose/docker-compose.complete.yml restart vllm
```

## Performance Tips

1. **Single GPU**: Set `VLLM_TENSOR_PARALLEL_SIZE=1` for best single-GPU performance
2. **Multi-GPU**: Increase `VLLM_TENSOR_PARALLEL_SIZE` to use multiple GPUs
3. **Memory**: Adjust `VLLM_GPU_MEMORY_UTILIZATION` based on available VRAM
4. **Throughput**: For high throughput, use smaller models or increase quantization

## Requirements

- NVIDIA GPU with CUDA support (Ampere architecture or newer recommended)
- CUDA Driver 535 or above
- Docker with NVIDIA Container Toolkit
- At least 8GB VRAM for default model
- HuggingFace token for gated models (optional, cached in `~/.cache/huggingface`)

## Troubleshooting

### Check Service Status
```bash
# View logs
docker compose -f deploy/compose/docker-compose.complete.yml logs -f vllm

# Check health
curl http://localhost:8001/v1/models
```

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check vLLM container GPU access
docker exec vllm-service nvidia-smi
```

### Model Loading Issues
- Ensure sufficient VRAM for the model
- Check HuggingFace cache: `ls ~/.cache/huggingface/hub`
- For gated models, set HF_TOKEN environment variable

## Comparison with Ollama

| Feature | Ollama | vLLM |
|---------|--------|------|
| **Ease of Use** | ✅ Very easy | ⚠️ More complex |
| **Model Management** | ✅ Built-in pull/push | ❌ Manual download |
| **Throughput** | ⚠️ Moderate | ✅ High |
| **Quantization** | Q4_K_M | FP8, GPTQ |
| **Memory Efficiency** | ✅ Good | ✅ Excellent (PagedAttention) |
| **Use Case** | Development, small-scale | Production, high-throughput |

## When to Use vLLM

Use vLLM when:
- Processing large batches of requests
- Need maximum throughput
- Using multiple GPUs
- Deploying to production with high load

Use Ollama when:
- Getting started with the project
- Single-user development
- Simpler model management needed
- Don't need maximum performance
