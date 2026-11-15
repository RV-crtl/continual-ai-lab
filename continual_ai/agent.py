"""Learning controller: propose, attempt, independently verify, persist."""
from __future__ import annotations
from .ollama_client import OllamaClient
from .storage import record
from .tasks import task_for_seed, instruction, verify


def learning_round(store, client: OllamaClient, *, use_memory: bool = True) -> dict:
    seed = store.get_next_seed()
    task = task_for_seed(seed)
    examples = store.examples(task.kind) if use_memory else []
    response = client.generate(instruction(task, examples), max_tokens=192)
    correct = verify(task, response)
    row = record(task, response, client.model, correct)
    store.add(row)
    return row


def evaluate(client: OllamaClient, n: int = 12, store=None) -> dict:
    """Held-out disjoint seed range; results are never fed to the training corpus."""
    if n < 1 or n > 500:
        raise ValueError("benchmark task count must be between 1 and 500")
    wins = 0
    per_kind = {}
    for ix in range(n):
        task = task_for_seed(9_000_000 + ix)
        response = client.generate(instruction(task, None))
        ok = verify(task, response)
        wins += int(ok)
        bucket = per_kind.setdefault(task.kind, {"passed": 0, "total": 0})
        bucket["passed"] += int(ok)
        bucket["total"] += 1
    return {"model": client.model, "n": n, "passed": wins,
            "accuracy": wins / n, "kinds": per_kind,
            "notes": "Held-out generated arithmetic; not an intelligence benchmark"}
