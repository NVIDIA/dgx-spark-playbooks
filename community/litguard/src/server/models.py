"""Model loading and inference logic for litguard."""

import os
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.environ.get(
            "LITGUARD_CONFIG",
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


# Label normalization: map various HF label schemes to injection/benign
INJECTION_LABELS = {"INJECTION", "LABEL_1", "injection", "1"}
BENIGN_LABELS = {"LEGIT", "LABEL_0", "SAFE", "benign", "legitimate", "0"}


def normalize_label(raw_label: str) -> str:
    if raw_label.upper() in {l.upper() for l in INJECTION_LABELS}:
        return "injection"
    return "benign"


class ModelInstance:
    def __init__(self, name: str, hf_model: str, device: str, batch_size: int):
        self.name = name
        self.hf_model = hf_model
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_model)
        if self.device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
        self.model.eval()
        # Build id2label map
        self.id2label = self.model.config.id2label

    def predict(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(texts)):
            predicted_id = torch.argmax(probs[i]).item()
            raw_label = self.id2label[predicted_id]
            label = normalize_label(raw_label)
            score = probs[i][predicted_id].item()
            results.append(
                {"label": label, "score": round(score, 4), "confidence": round(score, 4)}
            )
        return results


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, ModelInstance] = {}

    def load_from_config(self, config: dict):
        device_override = os.environ.get("DEVICE")
        for model_cfg in config.get("models", []):
            device = device_override or model_cfg.get("device", "cpu")
            instance = ModelInstance(
                name=model_cfg["name"],
                hf_model=model_cfg["hf_model"],
                device=device,
                batch_size=model_cfg.get("batch_size", 32),
            )
            instance.load()
            self.models[instance.name] = instance

    def get_default(self) -> ModelInstance:
        return next(iter(self.models.values()))

    def get(self, name: str) -> ModelInstance | None:
        return self.models.get(name)

    def list_models(self) -> list[dict]:
        return [
            {
                "name": m.name,
                "hf_model": m.hf_model,
                "device": m.device,
                "batch_size": m.batch_size,
            }
            for m in self.models.values()
        ]
