"""Read-only inference diagnostics. No training, no local-data modification."""
from __future__ import annotations
import argparse
import json
from .ollama_client import OllamaClient
from .tasks import task_for_seed, extract_integer_answer


def main(argv=None):
    p = argparse.ArgumentParser(description='Test model chat response before running autonomous learning')
    p.add_argument('--model', default='qwen3:4b')
    p.add_argument('--endpoint', default='http://127.0.0.1:11434')
    a = p.parse_args(argv)
    client = OllamaClient(a.model, a.endpoint)
    print('Model:', a.model, '\nEndpoint:', a.endpoint)
    print('Installed models:', client.models())
    parseable = 0
    for seed in (2, 3, 4):
        task = task_for_seed(seed)
        prompt = 'Solve the arithmetic problem and return ONLY the integer answer.\n' + task.question
        try:
            output = client.generate(prompt, temperature=0, max_tokens=192)
        except RuntimeError as e:
            print('INFERENCE ERROR:', e)
            return 2
        answer, parser = extract_integer_answer(output)
        parseable += int(answer is not None)
        print(json.dumps({
            'seed':seed, 'question':task.question, 'expected':task.expected,
            'output':output[:240], 'parser':parser, 'parsed':answer,
            'correct':answer==task.expected, 'ollama':client.last_metadata,
        }, ensure_ascii=False))
    print(f'Parseable answers: {parseable}/3')
    if parseable < 2:
        print('FAILED diagnostic. Do NOT resume autonomous learning. Check Ollama version/model.')
        return 1
    print('Inference is producing parsable answers. Inspect correctness and continue with a small run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
