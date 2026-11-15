"""Resource-limited QLoRA fine-tuning using independent deterministic training labels.

A new adapter is always a CANDIDATE. This module never overwrites the incumbent.
On NVIDIA Blackwell use a compatible CUDA-enabled PyTorch wheel and bitsandbytes.
The baseline model is reloaded afresh for each candidate (not adapter stacking).
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def get_dataset(path: Path, max_examples: int) -> list[dict]:
    from .storage import read_jsonl
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        if row.get('label_source') != 'deterministic_generator' or row.get('verified') is not True:
            continue
        seed = int(row['seed'])
        if not 0 <= seed < 9_000_000:
            continue
        # Never trust a copied or tampered 'verified' flag alone. Recompute label.
        from .tasks import task_for_seed
        t = task_for_seed(seed)
        if row.get('question') != t.question or int(row.get('expected')) != t.expected or row.get('kind') != t.kind:
            continue
        out[seed] = row
    return [out[k] for k in sorted(out)[-max_examples:]]


def prefix_for(tok, question: str) -> str:
    msgs = [
        {'role': 'system', 'content': 'You are a careful mathematical solver. Reply with exactly ANSWER: <integer>.'},
        {'role': 'user', 'content': question},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_features(tok, rows: list[dict], max_length: int) -> list[dict]:
    result = []
    for sample in rows:
        prefix = prefix_for(tok, sample['question'])
        pre = tok(prefix, add_special_tokens=False)['input_ids']
        completion = tok('ANSWER: ' + str(sample['expected']) + (tok.eos_token or ''),
                         add_special_tokens=False)['input_ids']
        ids = (pre + completion)[:max_length]
        labels = ([-100] * len(pre) + completion)[:max_length]
        if any(n != -100 for n in labels):
            result.append({'input_ids': ids, 'attention_mask': [1]*len(ids), 'labels': labels})
    return result


def train(dataset: Path, base_model: str, output_dir: Path, max_examples: int,
          max_steps: int, learning_rate: float = 0.0001, max_length: int = 256):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise RuntimeError('Install CUDA PyTorch, transformers, peft, accelerate and bitsandbytes in .venv-train') from error
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing expensive CPU fine-tuning')
    if max_steps < 1 or max_steps > 150:
        raise ValueError('Max training steps must be between 1 and 150')
    if not 32 <= max_examples <= 5000 or not 64 <= max_length <= 1024:
        raise ValueError('Invalid training data or token bounds')
    samples = get_dataset(dataset, max_examples)
    if len(samples) < 96:
        raise RuntimeError(f'Only {len(samples)} verified, rechecked examples; need at least 96')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite an existing candidate adapter')
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, trust_remote_code=False, quantization_config=bnb,
        device_map={'':0}, torch_dtype=dtype)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj','v_proj'], bias='none'))
    features = build_features(tok, samples, max_length)
    if len(features) < 96:
        raise RuntimeError('Too few usable training examples')

    class Data(torch.utils.data.Dataset):
        def __len__(self): return len(features)
        def __getitem__(self, i): return features[i]

    def collate(batch):
        length = max(len(row['input_ids']) for row in batch)
        return {
            'input_ids': torch.tensor([r['input_ids']+[tok.pad_token_id]*(length-len(r['input_ids'])) for r in batch]),
            'attention_mask': torch.tensor([r['attention_mask']+[0]*(length-len(r['attention_mask'])) for r in batch]),
            'labels': torch.tensor([r['labels']+[-100]*(length-len(r['labels'])) for r in batch]),
        }

    options = TrainingArguments(
        output_dir=str(output_dir), max_steps=max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=learning_rate, logging_steps=10,
        save_strategy='no', report_to='none', remove_unused_columns=False,
        gradient_checkpointing=True, bf16=dtype==torch.bfloat16,
        fp16=dtype==torch.float16, optim='adamw_torch',
        dataloader_num_workers=0, seed=42, max_grad_norm=1.0,
    )
    Trainer(model=model, args=options, train_dataset=Data(), data_collator=collate).train()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    manifest = {'base_model': base_model, 'verified_examples': len(features),
                'max_steps': max_steps, 'learning_rate': learning_rate,
                'last_training_seed': max(x['seed'] for x in samples),
                'dataset_sha256': digest,
                'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'status': 'UNPROMOTED_CANDIDATE',
                'scope': 'Elementary arithmetic, not general intelligence'}
    (output_dir/'TRAINING_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Candidate adapter saved; independent benchmark and promotion gate required.', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='QLoRA candidate training; no automatic promotion here')
    p.add_argument('--dataset', type=Path, default=Path('data/local/verified_sft.jsonl'))
    p.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-examples', type=int, default=512)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--learning-rate', type=float, default=0.0001)
    a = p.parse_args(argv)
    train(a.dataset, a.base_model, a.output_dir, a.max_examples, a.max_steps, a.learning_rate)


if __name__ == '__main__':
    main()
