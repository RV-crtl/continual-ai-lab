"""Dependency-free client for an Ollama instance on the same machine."""
from __future__ import annotations
import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434", timeout: int = 180):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def _json(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.endpoint + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as e:
            raise RuntimeError(
                f"Cannot use Ollama at {self.endpoint}. Check Ollama is running "
                f"and that you pulled the model '{self.model}'. Details: {e}"
            ) from e

    def models(self) -> list[str]:
        return [entry.get("name", "") for entry in self._json("/api/tags").get("models", [])]

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 192) -> str:
        data = self._json("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "keep_alive": "60s",
        })
        return str(data.get("response", ""))
