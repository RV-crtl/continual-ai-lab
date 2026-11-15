"""Deterministic mathematics curriculum and independently checked answer parsing.

Output parsing is separate from correctness. The mathematical ground truth always
comes from task_for_seed, never from an LLM's own evaluation.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import random
import re

ANSWER_RE = re.compile(r'(?im)^\s*ANSWER\s*:\s*(-?\d+)\s*$')
INTEGER_RE = re.compile(r'^\s*(-?\d+)\s*$')
FINAL_RE = re.compile(r'(?i)^\s*(?:final answer\s*[:=]|(?:the\s+)?answer\s+is)\s*(-?\d+)\s*\.?\s*$')
EQUATION_RE = re.compile(r'=\s*(-?\d+)\s*$')
VARIABLE_RE = re.compile(r'(?i)^\s*x\s*=\s*(-?\d+)\s*$')


@dataclass(frozen=True)
class Task:
    seed: int
    kind: str
    question: str
    expected: int


def task_for_seed(seed: int) -> Task:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = random.Random(seed ^ 0x5EF4CC15)
    kind = ['addition', 'multiplication', 'division', 'remainder', 'linear', 'powers'][seed % 6]
    if kind == 'addition':
        a, b = rng.randint(-1500, 1500), rng.randint(-1500, 1500)
        question, answer = f'Calculate ({a}) + ({b}).', a + b
    elif kind == 'multiplication':
        a, b = rng.randint(-75, 75), rng.randint(-75, 75)
        question, answer = f'Calculate ({a}) times ({b}).', a * b
    elif kind == 'division':
        divisor, quotient = rng.randint(2, 45), rng.randint(-50, 50)
        question, answer = f'Calculate ({divisor * quotient}) divided by ({divisor}).', quotient
    elif kind == 'remainder':
        n, d = rng.randint(200, 5000), rng.randint(2, 55)
        question, answer = f'Find the nonnegative remainder when {n} is divided by {d}.', n % d
    elif kind == 'linear':
        x, a, b = rng.randint(-60, 60), rng.choice([n for n in range(-12, 13) if n]), rng.randint(-90, 90)
        question, answer = f'Solve for integer x: ({a})*x + ({b}) = {a * x + b}.', x
    else:
        a, b = rng.randint(2, 13), rng.randint(2, 13)
        question, answer = f'Calculate {a} squared minus {b} squared.', a*a - b*b
    return Task(seed, kind, question, answer)


def extract_integer_answer(response: str) -> tuple[int | None, str]:
    """Recognise conservative, unambiguous final integer answers.

    Accepts a labeled answer, a standalone integer, a JSON integer answer, or
    a final arithmetic equation. Never selects a number just because it matches
    the expected answer. Multiple labeled answers are rejected.
    """
    if not isinstance(response, str) or not response.strip():
        return None, 'empty'
    value = response.strip()
    matches = ANSWER_RE.findall(value)
    if len(matches) > 1:
        return None, 'ambiguous'
    if len(matches) == 1:
        # Only a single explicit labeled answer is recognized; it can follow work.
        return int(matches[0]), 'labelled'
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and set(parsed) == {'answer'}:
        candidate = parsed['answer']
        if type(candidate) is int:
            return candidate, 'json'
        return None, 'unparseable'
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    last = lines[-1]
    single = INTEGER_RE.fullmatch(last)
    if single:
        return int(single.group(1)), 'plain_integer'
    variable = VARIABLE_RE.fullmatch(last)
    if variable:
        return int(variable.group(1)), 'variable_solution'
    explicit = FINAL_RE.fullmatch(last)
    if explicit:
        return int(explicit.group(1)), 'final_statement'
    # Only final equations with arithmetic context, no prose/answer placeholders.
    equation = EQUATION_RE.search(last)
    if equation and len(last) < 300 and not re.search(r'[A-Za-z<>]', last) and any(op in last for op in '+-*/%×÷^²³'):
        return int(equation.group(1)), 'final_equation'
    return None, 'unparseable'


def parse_integer_answer(response: str) -> int | None:
    return extract_integer_answer(response)[0]


def verify(task: Task, response: str) -> bool:
    return parse_integer_answer(response) == task.expected


def instruction(task: Task, examples: list[dict] | None = None) -> str:
    prefix = (
        'Solve the current arithmetic question carefully. Double-check the calculation. '
        'Return only the final integer, preferably using the exact format ANSWER: <number>. '
        'Use the actual computed number, never a placeholder or a copied example.\n'
        'Format examples (NOT the question to solve):\n'
        'Question: Calculate (3) + (4).\nANSWER: 7\n'
        'Question: Calculate (-2) times (6).\nANSWER: -12\n'
    )
    if examples:
        prefix += '\nPreviously independently checked examples (not the current question):\n'
        for example in examples[:2]:
            prefix += f"Question: {example['question']}\nANSWER: {example['expected']}\n"
    return prefix + '\nSolve the NEW question, without reusing a demonstration answer.\nQuestion: ' + task.question + '\n'
