"""GPU-governed continual training, paired verification and atomic promotion.

Run from a SEPARATE .venv-train on the physical NVIDIA Windows machine.
Training and comparison use separate processes to release VRAM between phases.
User Ctrl+C stops it. A lock pauses multi-agent learning while training runs.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .storage import LocalStore, append_jsonl
from .training_registry import promote, read_active


ROOT = Path('data/local')


def gpu_status() -> dict:
    if not shutil.which('nvidia-smi'):
        return {'ok': False, 'reason': 'nvidia-smi missing'}
    try:
        cmd = ['nvidia-smi', '--query-gpu=memory.total,memory.free,temperature.gpu,utilization.gpu',
               '--format=csv,noheader,nounits']
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8)
        values = [int(s.strip()) for s in r.stdout.splitlines()[0].split(',')]
        total, free, temp, utilization = values
        return {'ok': free >= 7500 and temp < 79 and utilization < 50,
                'total_mib': total, 'free_mib': free,
                'temp_c': temp, 'gpu_util_pct': utilization,
                'reason': ('GPU available' if free >= 7500 and temp < 79 and utilization < 50
                           else 'Waiting for >=7.5 GiB free VRAM, GPU <79 C, utilisation <50%')}
    except (OSError, ValueError, subprocess.SubprocessError, IndexError) as e:
        return {'ok': False, 'reason': f'GPU query failed: {e}'}


def run_command(args: list[str], timeout_seconds: int) -> None:
    print('Executing:', ' '.join(args), flush=True)
    subprocess.run(args, check=True, timeout=timeout_seconds)


def save_state(path: Path, data: dict) -> None:
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    os.replace(temp, path)


def cycle(root: Path, db: Path, base_model: str, min_examples: int,
          min_new: int, max_steps: int, max_samples: int, test_count: int,
          max_seconds: int, stop_on_error: bool = False) -> str:
    root.mkdir(parents=True, exist_ok=True)
    state_file = root / 'train_state.json'
    state = json.loads(state_file.read_text(encoding='utf-8')) if state_file.exists() else {}
    store = LocalStore(db)
    try:
        samples = store.training_examples()
    finally:
        store.close()
    last_seed = int(state.get('last_trained_seed', -1))
    new = sum(int(row['seed'] > last_seed) for row in samples)
    print(f'Available verified examples: {len(samples)}, new since last cycle: {new}', flush=True)
    if len(samples) < min_examples or new < min_new:
        return 'insufficient_new_data'
    status = gpu_status()
    print('Resource governor:', json.dumps(status), flush=True)
    if not status['ok']:
        return 'resource_deferred'
    incumbent = read_active(root)
    if incumbent and incumbent['base_model'] != base_model:
        raise ValueError('Active adapter base model differs; no mixed-model comparison permitted')
    # Do not share one GPU between unrestricted tool-calls and training.
    lock = root.parent / 'training.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'pid': os.getpid(), 'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}))
    except FileExistsError:
        return 'training_busy'
    label = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')
    candidate = root / 'candidates' / f'candidate-{label}'
    report_file = root / 'reports' / f'comparison-{label}.json'
    dataset = root / f'dataset-{label}.jsonl'
    try:
        # Verify training rows *again* in train_lora.get_dataset before fitting.
        with dataset.open('w', encoding='utf-8') as f:
            for row in samples:
                f.write(json.dumps(row, sort_keys=True) + '\n')
        run_command([sys.executable, '-m', 'continual_ai.train_lora', '--dataset', str(dataset),
                     '--base-model', base_model, '--output-dir', str(candidate),
                     '--max-examples', str(max_samples), '--max-steps', str(max_steps)], max_seconds)
        compare_cmd = [sys.executable, '-m', 'continual_ai.benchmark_adapter',
            '--base-model', base_model, '--candidate-adapter', str(candidate),
            '--n', str(test_count), '--out', str(report_file)]
        if incumbent:
            compare_cmd.extend(['--incumbent-adapter', incumbent['adapter']])
        run_command(compare_cmd, max_seconds)
        report = json.loads(report_file.read_text(encoding='utf-8'))
        if report['gate_passed']:
            promoted = promote(report_file, root)
            outcome = 'promoted'
            print('Validated candidate promoted:', promoted['adapter'], flush=True)
        else:
            outcome = 'rejected'
            print('Candidate rejected:', report['gate_reasons'], flush=True)
        state['last_trained_seed'] = max(row['seed'] for row in samples)
        state['last_outcome'] = outcome
        state['last_cycle_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_state(state_file, state)
        append_jsonl(root/'cycles.jsonl', {'cycle': label, 'outcome': outcome,
            'new_examples': new, 'candidate': str(candidate), 'report': str(report_file)})
        return outcome
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError, ValueError, OSError) as ex:
        append_jsonl(root/'cycles.jsonl', {'cycle': label, 'outcome': 'failed', 'error': str(ex)})
        print('Training cycle FAILED, incumbent unchanged:', str(ex), flush=True)
        if stop_on_error:
            raise
        return 'failed'
    finally:
        lock.unlink(missing_ok=True)
        dataset.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description='Train, double-benchmark, and promote only verified improvements')
    p.add_argument('--once', action='store_true')
    p.add_argument('--db', type=Path, default=ROOT/'learning.sqlite3')
    p.add_argument('--root', type=Path, default=ROOT/'training')
    p.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--min-examples', type=int, default=96)
    p.add_argument('--min-new', type=int, default=96)
    p.add_argument('--max-steps', type=int, default=40)
    p.add_argument('--max-samples', type=int, default=512)
    p.add_argument('--test-count', type=int, default=60)
    p.add_argument('--max-seconds', type=int, default=3600)
    p.add_argument('--check-every', type=int, default=600)
    a = p.parse_args(argv)
    if a.min_examples < 96 or a.min_new < 32 or a.test_count < 60 or a.check_every < 30:
        p.error('Minimum data and independent verification constraints cannot be reduced')
    if a.max_steps < 1 or a.max_steps > 150 or a.max_samples < 96:
        p.error('Invalid training limits')
    print('Training governor active. Ctrl+C stops after a current subprocess stops.', flush=True)
    try:
        while True:
            result = cycle(a.root, a.db, a.base_model, a.min_examples, a.min_new,
                           a.max_steps, a.max_samples, a.test_count, a.max_seconds)
            print('Cycle status:', result, flush=True)
            if a.once:
                break
            time.sleep(a.check_every)
    except KeyboardInterrupt:
        print('Training supervisor stopped by user.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
