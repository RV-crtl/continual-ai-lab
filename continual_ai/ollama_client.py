"""Native Ollama chat client with bounded, inspectable response handling.

For verifiable arithmetic use schema-constrained answers and refuse partial
responses. Never fabricate a result from a task oracle or model thinking trace.
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request

ANSWER_SCHEMA = {
    "type": "object", "properties": {"answer": {"type": "integer"}},
    "required": ["answer"], "additionalProperties": False
}

class OllamaClient:
    def __init__(self, model: str, endpoint: str = "http://127.0.0.1:11434", timeout: int = 180):
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.last_metadata = {}

    def _json(self, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.endpoint + path, data=data, headers={"Content-Type": "application/json"},
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

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256,
                 answer_mode: bool = False) -> str:
        """Require complete, parseable answers for narrow arithmetic tasks."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        from .tasks import extract_integer_answer
        for attempt, tokens in enumerate((max_tokens, min(max(max_tokens * 2, 512), 1024))):
            if answer_mode:
                messages = [
                    {"role": "system", "content": (
                        "You are an arithmetic solver. Return only a JSON object with one integer "
                        "property named answer. Do not explain the calculation. Never copy examples."
                    )},
                    {"role": "user", "content": prompt.rstrip() + "\n/no_think"},
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
            request = {
                "model": self.model,
                "messages": messages, "stream": False, "think": False,
                "options": {"temperature": temperature, "num_predict": tokens},
                "keep_alive": "60s",
            }
            if answer_mode:
                request["format"] = ANSWER_SCHEMA
            data = self._json("/api/chat", request)
            message = data.get("message", {})
            if not isinstance(message, dict):
                raise RuntimeError("Ollama /api/chat did not return a message object")
            output = message.get("content", "")
            reason = data.get("done_reason")
            parsed, mode = extract_integer_answer(output) if isinstance(output, str) else (None, "invalid")
            self.last_metadata = {
                "endpoint": "/api/chat", "model": self.model,
                "eval_count": data.get("eval_count"), "done_reason": reason,
                "thinking_present": bool(message.get("thinking")),
                "answer_length": len(output) if isinstance(output, str) else -1,
                "parser": mode if answer_mode else None, "attempt": attempt + 1,
            }
            complete = reason != "length" and isinstance(output, str) and bool(output.strip())
            if complete and (not answer_mode or parsed is not None):
                return output
        info = self.last_metadata
        raise RuntimeError(
            f"Inference did not produce a complete, parseable final answer for {self.model} "
            f"after two attempts (reason={info.get('done_reason')}, "
            f"generated={info.get('eval_count')}, parser={info.get('parser')}). "
            "Do not resume learning. Try: ollama pull qwen3:8b; "
            "python -m continual_ai.inference_doctor --model qwen3:8b"
        )
