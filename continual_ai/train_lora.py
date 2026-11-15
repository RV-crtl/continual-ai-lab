"""Real, OPTIONAL LoRA weight update from independently labelled samples.

Requires `pip install -r requirements-train.txt`. Best on an NVIDIA GPU.
This writes a candidate adapter. It does NOT change the active Ollama model.
A separate held-out benchmark is required before promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def get_dataset(path: Path, max_examples: int) -> list[dict]:
    from .storage import read_jsonl
    rows = read_jsonl(path)
    out = []
    seeds = set()
    for row in rows:
        if row.get("label_source") != "deterministic_generator" or not row.get("verified"):
            continue
        seed = int(row["seed"])
        if seed >= 9_000_000 or seed in seeds:
            continue
        seeds.add(seed)
        out.append(row)
        if len(out) >= max_examples:
            break
    return out


def prefix_for(tok, question: str) -> str:
    msgs = [
        {"role": "system", "content": "You are a careful mathematical solver. Reply with exactly ANSWER: <integer>."},
        {"role": "user", "content": question},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_features(tok, rows: list[dict], max_length: int) -> list[dict]:
    result = []
    for sample in rows:
        prefix = prefix_for(tok, sample["question"])
        pre = tok(prefix, add_special_tokens=False)["input_ids"]
        completion = tok("ANSWER: " + str(sample["expected"]) + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
        ids = (pre + completion)[:max_length]
        labels = ([-100] * len(pre) + completion)[:max_length]
        if any(n != -100 for n in labels):
            result.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})
    return result


def train(dataset: Path, base_model: str, output_dir: Path, max_examples: int, epochs: float):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("Training dependencies missing. Run pip install -r requirements-train.txt") from error
    samples = get_dataset(dataset, max_examples)
    if len(samples) < 32:
        raise RuntimeError(f"Only {len(samples)} verified examples; at least 32 required.")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=False,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    model = get_peft_model(model, config)
    features = build_features(tok, samples, max_length=384)
    if len(features) < 32:
        raise RuntimeError("Insufficient examples survived tokenization.")

    class FeatureSet(torch.utils.data.Dataset):
        def __len__(self):
            return len(features)
        def __getitem__(self, index):
            return features[index]

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        return {
            "input_ids": torch.tensor([row["input_ids"] + [tok.pad_token_id] * (length - len(row["input_ids"])) for row in batch]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * (length - len(row["attention_mask"])) for row in batch]),
            "labels": torch.tensor([row["labels"] + [-100] * (length - len(row["labels"])) for row in batch]),
        }

    settings = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=0.0002, logging_steps=10,
        save_strategy="no", report_to="none", remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=settings, train_dataset=FeatureSet(), data_collator=collate)
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    output_dir.joinpath("TRAINING_MANIFEST.json").write_text(json.dumps({
        "base_model": base_model, "verified_examples": len(features),
        "epochs": epochs, "status": "UNPROMOTED_CANDIDATE",
        "warning": "Not automatically adopted; compare on separate held-out tasks first."
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate adapter saved to {output_dir}; NOT automatically promoted.")


def main():
    p = argparse.ArgumentParser(description="Train a candidate LoRA adapter from verified examples")
    p.add_argument("--dataset", type=Path, default=Path("data/local/verified_sft.jsonl"))
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", type=Path, default=Path("data/local/candidates/adapter-v1"))
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--epochs", type=float, default=1.0)
    args = p.parse_args()
    train(args.dataset, args.base_model, args.output_dir, args.max_examples, args.epochs)


if __name__ == "__main__":
    main()
