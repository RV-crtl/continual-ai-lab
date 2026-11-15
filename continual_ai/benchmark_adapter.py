"""Compare pretrained base versus candidate LoRA with independent test labels.

A passing verdict authorizes HUMAN review, not automatic promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .tasks import parse_integer_answer, task_for_seed
from .train_lora import prefix_for


def test_model(model, tok, n: int) -> dict:
    import torch
    model.eval()
    wins = 0
    items = []
    for i in range(n):
        task = task_for_seed(9_000_000 + i)
        prompt = prefix_for(tok, task.question)
        tokenized = tok(prompt, return_tensors="pt", add_special_tokens=False)
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        with torch.no_grad():
            generated = model.generate(**tokenized, do_sample=False, max_new_tokens=96,
                                       pad_token_id=tok.pad_token_id)
        output = tok.decode(generated[0, tokenized["input_ids"].shape[1]:], skip_special_tokens=True)
        correct = parse_integer_answer(output) == task.expected
        wins += int(correct)
        items.append({"seed": task.seed, "kind": task.kind, "correct": correct, "answer": output[:150]})
        print(f"[{'PASS' if correct else 'FAIL'}] {i+1}/{n} {task.kind}", flush=True)
    return {"n": n, "correct": wins, "accuracy": wins/n, "items": items}


def compare(base_model: str, adapter: Path, n: int, outfile: Path):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if n < 12:
        raise ValueError("At least 12 held-out tests required")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    def fresh_base():
        return AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(
            "cuda" if torch.cuda.is_available() else "cpu")
    model = fresh_base()
    print("Testing base model")
    baseline = test_model(model, tok, n)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Testing candidate adapter")
    model = PeftModel.from_pretrained(fresh_base(), str(adapter), is_trainable=False)
    candidate = test_model(model, tok, n)
    improvement = candidate["correct"] - baseline["correct"]
    # Require 2 additional successes on >=12 trials; still insufficient for broad claims.
    verdict = {"base_model": base_model, "adapter": str(adapter), "baseline": baseline,
               "candidate": candidate, "pass": improvement >= 2,
               "correct_improvement": improvement,
               "status": "REVIEW_CANDIDATE" if improvement >= 2 else "REJECT_CANDIDATE",
               "note": "Narrow small-sample benchmark; review wider skills and regressions manually."}
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation: {verdict['status']} | {outfile}")


def main():
    p = argparse.ArgumentParser(description="Independent base-vs-LoRA benchmark")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("data/local/adapter_comparison.json"))
    args = p.parse_args()
    compare(args.base_model, args.adapter, args.n, args.out)


if __name__ == "__main__":
    main()
