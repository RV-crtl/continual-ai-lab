"""Deterministic, independently-verifiable mathematics curriculum.

This intentionally narrow domain is a *proof of learning infrastructure*, not
an adequate benchmark of general intelligence.
"""
from __future__ import annotations
from dataclasses import dataclass
import random
import re

ANSWER_RE = re.compile(r"(?im)^\s*ANSWER:\s*(-?\d+)\s*$")

@dataclass(frozen=True)
class Task:
    seed: int
    kind: str
    question: str
    expected: int


def task_for_seed(seed: int) -> Task:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = random.Random(seed ^ 0x5EF4CC15)
    kind = ["addition", "multiplication", "division", "remainder", "linear", "powers"][seed % 6]
    if kind == "addition":
        a, b = rng.randint(-1500, 1500), rng.randint(-1500, 1500)
        question, answer = f"Calculate ({a}) + ({b}).", a + b
    elif kind == "multiplication":
        a, b = rng.randint(-75, 75), rng.randint(-75, 75)
        question, answer = f"Calculate ({a}) times ({b}).", a * b
    elif kind == "division":
        divisor, quotient = rng.randint(2, 45), rng.randint(-50, 50)
        question, answer = f"Calculate ({divisor * quotient}) divided by ({divisor}).", quotient
    elif kind == "remainder":
        n, d = rng.randint(200, 5000), rng.randint(2, 55)
        question, answer = f"Find the nonnegative remainder when {n} is divided by {d}.", n % d
    elif kind == "linear":
        x, a, b = rng.randint(-60, 60), rng.choice([n for n in range(-12, 13) if n]), rng.randint(-90, 90)
        question, answer = f"Solve for integer x: ({a})*x + ({b}) = {a * x + b}.", x
    else:
        a, b = rng.randint(2, 13), rng.randint(2, 13)
        question, answer = f"Calculate {a} squared minus {b} squared.", a*a - b*b
    return Task(seed, kind, question, answer)


def parse_integer_answer(response: str) -> int | None:
    matches = ANSWER_RE.findall(response)
    if len(matches) != 1:
        return None
    return int(matches[0])


def verify(task: Task, response: str) -> bool:
    return parse_integer_answer(response) == task.expected


def instruction(task: Task, examples: list[dict] | None = None) -> str:
    prefix = (
        "Solve the current arithmetic question and output ONLY its calculated integer result. "
        "Your reply MUST be exactly one line starting with ANSWER: followed by the actual number. "
        "Never print a placeholder, never copy an example answer, and do not add explanations.\n"
        "Format demonstrations (not the current task):\n"
        "Question: Calculate (3) + (4).\nANSWER: 7\n"
        "Question: Calculate (-2) times (6).\nANSWER: -12\n"
    )
    if examples:
        prefix += "\nPreviously independently verified examples (not the current question):\n"
        for example in examples[:2]:
            prefix += f"Question: {example['question']}\nANSWER: {example['expected']}\n"
    return prefix + "\nNow solve the NEW question; do not reuse a demonstrated answer.\nQuestion: " + task.question + "\n"
