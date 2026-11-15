"""Four roles: learner, strategy optimiser, independent verifier, resource governor.

Verification receives 2 of 4 *inference calls* per cycle (50% scheduled call
budget). This is NOT a claim about 50% GPU energy or wall-clock time; measured
runtime per role is reported to make the distinction explicit.

The verifier uses fresh, reserved tasks and the deterministic oracle; the model
is not allowed to decide whether its own answer is correct. Live verifier tasks
are never used for SFT or for the frozen promotion benchmark.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from .ollama_client import OllamaClient
from .tasks import task_for_seed, instruction, verify
from .storage import LocalStore, CloudStore, record, append_jsonl

VERIFIER_SEED_START = 9_200_000
STRATEGIES = ('few_shot', 'zero_shot', 'explicit_format')


def choose_strategy(counters: dict) -> str:
    """UCB selection. Optimiser may change prompts, never evaluation code."""
    total = sum(int(counters.get(s, {}).get('n', 0)) for s in STRATEGIES)
    for strategy in STRATEGIES:
        if int(counters.get(strategy, {}).get('n', 0)) == 0:
            return strategy
    return max(STRATEGIES, key=lambda s: (
        int(counters[s].get('correct', 0)) / int(counters[s]['n'])
        + math.sqrt(2 * math.log(total) / int(counters[s]['n']))
    ))


def strategy_prompt(task, examples, strategy):
    if strategy == 'few_shot':
        return instruction(task, examples)
    if strategy == 'zero_shot':
        return ('Answer the arithmetic problem. Reply on one line exactly as '
                'ANSWER: <computed integer>. No extra text.\nQuestion: '+task.question)
    if strategy == 'explicit_format':
        return ('You are checking elementary arithmetic. Compute exactly. '
                'Give one line only, e.g. ANSWER: 7 (the 7 is only an example). '
                'No explanation.\nCurrent question: '+task.question)
    raise ValueError('Unknown strategy')


@dataclass(frozen=True)
class CallBudget:
    learner: int = 1
    optimiser: int = 1
    verifier: int = 2

    def __post_init__(self):
        if min(self.learner, self.optimiser, self.verifier) <= 0:
            raise ValueError('All agent roles require positive inference allocations')
        if self.verifier * 2 < self.total:
            raise ValueError('Independent verification must receive at least 50% of inference calls')

    @property
    def total(self):
        return self.learner + self.optimiser + self.verifier

    @property
    def verification_share(self):
        return self.verifier / self.total


class MultiAgentController:
    def __init__(self, store, client, state_dir: Path, budget: CallBudget = CallBudget()):
        if budget != CallBudget():
            raise ValueError('Current orchestrator supports exactly 1:1:2 calls per cycle')
        self.store, self.client, self.budget = store, client, budget
        self.root = state_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'agent_state.json'
        self.log = self.root / 'agent_cycles.jsonl'
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {
            'cycles': 0, 'calls': {'learner': 0, 'optimiser': 0, 'verifier': 0},
            'seconds': {'learner': 0.0, 'optimiser': 0.0, 'verifier': 0.0},
            'strategies': {s: {'n': 0, 'correct': 0} for s in STRATEGIES},
            'verified_correct': 0,
        }

    def _run(self, task, role, prompt):
        started = time.monotonic()
        response = self.client.generate(prompt, temperature=0, max_tokens=80)
        elapsed = time.monotonic() - started
        # Only the deterministic task generator may determine the ground truth.
        ok = verify(task, response)
        self.state['calls'][role] += 1
        self.state['seconds'][role] += elapsed
        row = record(task, response, self.client.model, ok,
                     split='verify' if role == 'verifier' else 'learn')
        self.store.add(row)
        return ok, elapsed, row

    def cycle(self) -> dict:
        if (self.root / 'training.lock').exists():
            raise RuntimeError('A training run has reserved the GPU; retry after it finishes')
        seed = self.store.get_next_seed()
        if seed >= 9_000_000:
            raise ValueError('Training seeds exhausted before reserved benchmark boundary')
        task = task_for_seed(seed)
        baseline, learner_time, _ = self._run(task, 'learner', instruction(task, self.store.examples(task.kind)))
        seed = self.store.get_next_seed()
        if seed >= 9_000_000:
            raise ValueError('Training seed boundary reached')
        task = task_for_seed(seed)
        strategy = choose_strategy(self.state['strategies'])
        improved, optimiser_time, _ = self._run(task, 'optimiser',
            strategy_prompt(task, self.store.examples(task.kind), strategy))
        strategy_counts = self.state['strategies'][strategy]
        strategy_counts['n'] += 1
        strategy_counts['correct'] += int(improved)
        verified = 0
        verification_seconds = 0.0
        cycle_index = self.state['cycles']
        for j in range(self.budget.verifier):
            task = task_for_seed(VERIFIER_SEED_START + 2*cycle_index + j)
            ok, duration, _ = self._run(task, 'verifier', strategy_prompt(task, None, 'zero_shot'))
            verified += int(ok)
            verification_seconds += duration
        self.state['verified_correct'] += verified
        self.state['cycles'] += 1
        # Persist cycle metadata atomically; interrupted attempts remain in store.
        total_calls = sum(self.state['calls'].values())
        report = {'cycle': self.state['cycles'], 'strategy': strategy,
                  'learner_correct': bool(baseline), 'optimiser_correct': bool(improved),
                  'verifier_correct': verified, 'verifier_total': self.budget.verifier,
                  'verification_call_share': self.budget.verification_share,
                  'measured_seconds': {'learner': round(learner_time, 3),
                      'optimiser': round(optimiser_time, 3),
                      'verifier': round(verification_seconds, 3)},
                  'cumulative_calls': total_calls}
        append_jsonl(self.log, report)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.state, indent=2)+'\n', encoding='utf-8')
        os.replace(temp, self.path)
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description='Multi-agent local/cloud verified learning (50% verifier call budget)')
    p.add_argument('mode', choices=['local', 'cloud'])
    p.add_argument('--model', default=None)
    p.add_argument('--endpoint', default='http://127.0.0.1:11434')
    p.add_argument('--cycles', type=int, default=0, help='0 = run until stopped (local only)')
    p.add_argument('--pause', type=float, default=2.0)
    p.add_argument('--max-seconds', type=int, default=1100)
    p.add_argument('--state-dir', type=Path, default=None)
    p.add_argument('--db', type=Path, default=Path('data/local/learning.sqlite3'))
    a = p.parse_args(argv)
    if a.cycles < 0 or a.pause < 0 or a.max_seconds < 30:
        p.error('Invalid limits')
    if a.mode == 'cloud' and not 1 <= a.cycles <= 25:
        p.error('Cloud mode requires --cycles between 1 and 25')
    root = a.state_dir or Path('data/cloud' if a.mode == 'cloud' else 'data/local')
    model = a.model or ('qwen3:1.7b' if a.mode == 'cloud' else 'qwen3:4b')
    store = CloudStore(root) if a.mode == 'cloud' else LocalStore(a.db)
    client = OllamaClient(model, a.endpoint)
    controller = MultiAgentController(store, client, root)
    started = time.monotonic()
    count = 0
    try:
        while not a.cycles or count < a.cycles:
            if time.monotonic() - started >= a.max_seconds and a.mode == 'cloud':
                print('Cloud time budget reached; preserved completed cycles.', flush=True)
                break
            if (root / 'training.lock').exists():
                print('Training has reserved GPU; learning agents paused.', flush=True)
                time.sleep(15)
                continue
            report = controller.cycle()
            count += 1
            print(json.dumps(report, sort_keys=True), flush=True)
            if a.pause:
                time.sleep(a.pause)
    except KeyboardInterrupt:
        print('Stopped. Recorded learning and verification attempts are preserved.')
    finally:
        if a.mode == 'cloud':
            store.write_summary()
        else:
            store.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
