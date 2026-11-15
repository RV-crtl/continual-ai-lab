"""Ollama client with explicit non-thinking chat inference and failure diagnostics.

Qwen models may separate their thinking trace from final content. Never treat an
empty final response as a legitimate answer or save it as a failed maths task.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request


class OllamaClient:
    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434", timeout: int = 180):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.last_metadata = {}

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
        """Generate a final answer using /api/chat, with bounded empty-reply retry.

        `think` belongs at the top level. Do not pull a result from the thinking
        trace or guess a value when the answer field is blank.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        for attempt, tokens in enumerate((max_tokens, max(max_tokens * 2, 256))):
            data = self._json("/api/chat", {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"temperature": temperature, "num_predict": tokens},
                "keep_alive": "60s",
            })
            message = data.get("message", {})
            if not isinstance(message, dict):
                raise RuntimeError("Ollama /api/chat did not return a message object")
            output = message.get("content", "")
            self.last_metadata = {
                "endpoint": "/api/chat", "model": self.model,
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason"),
                "thinking_present": bool(message.get("thinking")),
                "answer_length": len(output) if isinstance(output, str) else -1,
                "attempt": attempt + 1,
            }
            if isinstance(output, str) and output.strip():
                return output
        info = self.last_metadata
        raise RuntimeError(
            "Ollama returned an empty FINAL answer twice for " + self.model +
            f" (reason={info.get('done_reason')}, generated={info.get('eval_count')} tokens, "
            f"thinking_present={info.get('thinking_present')}). "
            "Learning paused instead of logging thousands of empty failures. "
            "Run: python -m continual_ai.inference_doctor"
        )
