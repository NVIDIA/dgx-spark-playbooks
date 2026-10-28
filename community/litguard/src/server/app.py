"""LitServe app for litguard - prompt injection detection."""

import json
import time
import os
import subprocess

import litserve as ls
from fastapi.middleware.cors import CORSMiddleware

from .models import ModelRegistry, load_config
from .metrics import metrics, ClassificationRecord


class PromptInjectionAPI(ls.LitAPI):
    def setup(self, device: str):
        self.config = load_config()
        self.registry = ModelRegistry()
        self.registry.load_from_config(self.config)

    def decode_request(self, request: dict) -> dict:
        # Support OpenAI chat completions format
        messages = request.get("messages", [])
        model_name = request.get("model")
        # Extract text from the last user message
        text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle content array format
                    text = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                else:
                    text = content
                break
        return {"text": text, "model": model_name}

    def predict(self, inputs: dict) -> dict:
        text = inputs["text"]
        model_name = inputs.get("model")

        if model_name:
            model = self.registry.get(model_name)
        else:
            model = None

        if model is None:
            model = self.registry.get_default()

        start = time.time()
        results = model.predict([text])
        latency_ms = (time.time() - start) * 1000

        result = results[0]

        # Record metrics
        metrics.record(
            ClassificationRecord(
                timestamp=time.time(),
                input_text=text,
                model=model.name,
                label=result["label"],
                score=result["score"],
                latency_ms=latency_ms,
            )
        )

        return {**result, "model": model.name, "latency_ms": round(latency_ms, 2)}

    def encode_response(self, output: dict) -> dict:
        # Return as OpenAI-compatible chat completion response
        result_json = json.dumps(
            {
                "label": output["label"],
                "score": output["score"],
                "confidence": output["confidence"],
            }
        )
        return {
            "id": f"chatcmpl-litguard-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": output["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result_json},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def _get_gpu_utilization() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "N/A"


def create_app():
    config = load_config()
    api = PromptInjectionAPI()

    server = ls.LitServer(
        api,
        api_path="/v1/chat/completions",
        timeout=30,
    )

    # Build model info from config (available without worker process)
    model_info = [
        {
            "name": m["name"],
            "hf_model": m["hf_model"],
            "device": os.environ.get("DEVICE", m.get("device", "cpu")),
            "batch_size": m.get("batch_size", 32),
        }
        for m in config.get("models", [])
    ]
    model_names = [m["name"] for m in model_info]

    # Add custom endpoints via FastAPI app
    fastapi_app = server.app

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok", "models_loaded": model_names}

    @fastapi_app.get("/models")
    def list_models():
        return {"models": model_info}

    @fastapi_app.get("/metrics")
    def get_metrics():
        m = metrics.get_metrics()
        m["gpu_utilization"] = _get_gpu_utilization()
        m["models_loaded"] = model_info
        return m

    @fastapi_app.get("/api/history")
    def get_history():
        return {"history": metrics.get_history()}

    return server


if __name__ == "__main__":
    config = load_config()
    server = create_app()
    server.run(port=config.get("port", 8234), host="0.0.0.0")
