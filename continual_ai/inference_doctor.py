"""Read-only model quality gate; does not write training examples or change weights."""
from __future__ import annotations
import argparse
import json
import os
from .ollama_client import OllamaClient
from .tasks import task_for_seed, extract_integer_answer

SEEDS = (2, 3, 4, 5, 6, 7)

def main(argv=None):
    p = argparse.ArgumentParser(description="Verify arithmetic inference before starting agents")
    p.add_argument("--model", default=os.getenv("CAI_MODEL", "qwen3:8b"))
    p.add_argument("--endpoint", default="http://127.0.0.1:11434")
    p.add_argument("--samples", type=int, default=3, choices=(3, 6))
    a = p.parse_args(argv)
    client = OllamaClient(a.model, a.endpoint)
    print("Model:", a.model, "Endpoint:", a.endpoint, flush=True)
    installed = client.models()
    print("Installed:", installed, flush=True)
    if a.model not in installed:
        print("Model not installed. Run: ollama pull " + a.model, flush=True)
        return 2
    parsed_count = correct = 0
    for seed in SEEDS[:a.samples]:
        task = task_for_seed(seed)
        prompt = "Compute the integer answer to the following problem.\n" + task.question
        try:
            output = client.generate(prompt, temperature=0, max_tokens=256, answer_mode=True)
            answer, method = extract_integer_answer(output)
            metadata = client.last_metadata
        except RuntimeError as ex:
            print("INFERENCE ERROR:", ex, flush=True)
            return 2
        parsed_count += int(answer is not None)
        correct += int(answer == task.expected)
        print(json.dumps({"question": task.question, "expected": task.expected,
                          "response": output[:300], "parsed": answer,
                          "parser": method, "correct": answer == task.expected,
                          "ollama": metadata}, ensure_ascii=False), flush=True)
    print(f"Doctor: {correct}/{a.samples} correct; {parsed_count}/{a.samples} parseable", flush=True)
    if correct != a.samples:
        print("FAIL: keep agents and GPU training stopped. Test qwen3:8b or another instruct model.", flush=True)
        return 1
    print("PASS: complete arithmetic outputs. First run a SMALL 5-cycle agent test.", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
