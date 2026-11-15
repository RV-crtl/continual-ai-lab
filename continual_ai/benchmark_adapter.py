"""Two independent paired holdouts for incumbent/candidate adapter comparisons.

The frozen seed ranges are reserved in tasks.py's generator; neither is
exported for training. No sampling, no in-context examples, and same tokenizer.
Passing is still narrow arithmetic evidence, not a general intelligence claim.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
from .tasks import task_for_seed, parse_integer_answer
from .train_lora import prefix_for

SPLITS = {'primary': 9_000_000, 'audit': 9_100_000}


def test_model(model, tok, start_seed: int, count: int) -> dict:
    import torch
    model.eval()
    correct = 0
    kinds = {}
    samples = []
    for i in range(count):
        task = task_for_seed(start_seed + i)
        prompt = prefix_for(tok, task.question)
        batch = tok(prompt, return_tensors='pt', add_special_tokens=False)
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.inference_mode():
            generated = model.generate(**batch, do_sample=False, max_new_tokens=40,
                                       pad_token_id=tok.pad_token_id)
        output = tok.decode(generated[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
        ok = parse_integer_answer(output) == task.expected
        correct += int(ok)
        entry = kinds.setdefault(task.kind, {'passed': 0, 'total': 0})
        entry['passed'] += int(ok)
        entry['total'] += 1
        samples.append({'seed': task.seed, 'kind': task.kind, 'correct': bool(ok)})
    return {'n': count, 'correct': correct, 'accuracy': round(correct/count, 4),
            'kinds': kinds, 'samples': samples}


def compare(base_model: str, candidate_adapter: Path, incumbent_adapter: Path | None,
            n: int, outfile: Path):
    if not 60 <= n <= 360:
        raise ValueError('Require 60–360 paired tests per independent benchmark split')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for the comparison')
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    def load_model(adapter):
        base = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=False,
            quantization_config=config, device_map={'':0}, torch_dtype=dtype)
        return PeftModel.from_pretrained(base, str(adapter), is_trainable=False) if adapter else base

    baseline = {}
    model = load_model(incumbent_adapter)
    for name, seed in SPLITS.items():
        print(f'Incumbent: {name}', flush=True)
        baseline[name] = test_model(model, tok, seed, n)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = load_model(candidate_adapter)
    candidate = {}
    for name, seed in SPLITS.items():
        print(f'Candidate: {name}', flush=True)
        candidate[name] = test_model(model, tok, seed, n)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from .training_registry import evaluate_gate
    report = {
        'base_model': base_model,
        'candidate_adapter': str(candidate_adapter.resolve()),
        'incumbent_adapter': str(incumbent_adapter.resolve()) if incumbent_adapter else None,
        'primary': {'baseline': baseline['primary'], 'candidate': candidate['primary']},
        'audit': {'baseline': baseline['audit'], 'candidate': candidate['audit']},
        'note': 'Reserved paired arithmetic tests; not evidence of general intelligence',
    }
    passed, reasons = evaluate_gate(report)
    report['gate_passed'] = passed
    report['gate_reasons'] = reasons
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('PROMOTION ELIGIBLE' if passed else 'REJECTED', '; '.join(reasons), flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='Paired double-holdout candidate evaluation')
    p.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--candidate-adapter', type=Path, required=True)
    p.add_argument('--incumbent-adapter', type=Path)
    p.add_argument('--n', type=int, default=60)
    p.add_argument('--out', type=Path, required=True)
    a=p.parse_args(argv)
    compare(a.base_model, a.candidate_adapter, a.incumbent_adapter, a.n, a.out)


if __name__ == '__main__':
    main()
