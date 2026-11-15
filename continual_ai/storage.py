"""SQLite-backed local history; JSONL-backed online history.

A trusted label is computed from the task generator, never from an AI's
self-assessment. Records never contain local user chats in the cloud build.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sqlite3

from .tasks import extract_integer_answer, task_for_seed


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def record(task, response: str, model: str, correct: bool, split: str = "learn") -> dict:
    return {
        "time": stamp(), "seed": task.seed, "kind": task.kind,
        "question": task.question, "expected": task.expected,
        "model_response": response[:3000], "correct": bool(correct),
        "model": model, "split": split, "label_source": "deterministic_generator",
    }


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                result.append(json.loads(line))
    return result


class LocalStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT NOT NULL, seed INTEGER, kind TEXT, question TEXT,
            expected INTEGER, model_response TEXT, correct INTEGER,
            model TEXT, split TEXT, label_source TEXT
        );
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT,
            role TEXT, content TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_attempts_seed ON attempts(seed, split);
        """)
        self.db.commit()

    def add(self, row: dict) -> None:
        columns = "time,seed,kind,question,expected,model_response,correct,model,split,label_source"
        self.db.execute(
            "INSERT INTO attempts ("+columns+") VALUES (?,?,?,?,?,?,?,?,?,?)",
            [row[k] for k in columns.split(",")],
        )
        self.db.commit()

    def get_next_seed(self) -> int:
        value = self.db.execute("SELECT MAX(seed) FROM attempts WHERE split='learn'").fetchone()[0]
        return 0 if value is None else int(value) + 1

    def examples(self, kind: str, n: int = 2) -> list[dict]:
        # Recheck stored examples under the corrected parser and deterministic oracle.
        # Never promote an incorrect model answer merely because a database bit says 1.
        rows = self.db.execute(
            "SELECT seed,question,expected,model_response FROM attempts "
            "WHERE split='learn' AND kind=? ORDER BY id DESC LIMIT 100", (kind,),
        ).fetchall()
        results = []
        for seed, question, expected, response in rows:
            task = task_for_seed(int(seed))
            parsed, _ = extract_integer_answer(response)
            if task.question == question and task.expected == expected and parsed == task.expected:
                results.append({'question': task.question, 'expected': task.expected})
                if len(results) >= n:
                    break
        return results

    def stats(self) -> dict:
        """Re-score existing responses without modifying their historical records.

        A legacy failure caused by a formatting-only rejection is counted as a
        recovered success, separately from a genuine incorrect answer.
        """
        rows = self.db.execute(
            "SELECT seed,kind,question,expected,model_response,correct FROM attempts "
            "WHERE split='learn' ORDER BY id"
        ).fetchall()
        total = len(rows)
        passed = legacy_passed = rescued = unparseable = bad_label = 0
        by_kind = {}
        rolling = []
        for seed, kind, question, expected, response, old_correct in rows:
            legacy_passed += int(bool(old_correct))
            task = task_for_seed(int(seed))
            valid_label = kind == task.kind and question == task.question and expected == task.expected
            answer, parser = extract_integer_answer(response)
            correct = bool(valid_label and answer == task.expected)
            passed += int(correct)
            rescued += int(correct and not old_correct)
            unparseable += int(answer is None)
            bad_label += int(not valid_label)
            bucket = by_kind.setdefault(kind, {'total': 0, 'passed': 0})
            bucket['total'] += 1
            bucket['passed'] += int(correct)
            rolling.append(int(correct))
        window = rolling[-100:]
        return {'total': total, 'correct': passed,
                'accuracy': round(passed / total, 4) if total else None,
                'legacy_correct': legacy_passed, 'rescued_from_format': rescued,
                'unparseable': unparseable, 'invalid_stored_labels': bad_label,
                'recent_count': len(window), 'recent_correct': sum(window),
                'recent_accuracy': round(sum(window)/len(window), 4) if window else None,
                'kinds': by_kind}

    def recent_attempts(self, n: int = 12) -> list[dict]:
        rows = self.db.execute(
            "SELECT seed,kind,question,expected,model_response,correct,model FROM attempts "
            "WHERE split='learn' ORDER BY id DESC LIMIT ?", (n,),
        ).fetchall()
        result = []
        for seed, kind, question, expected, response, old_correct, model in rows:
            task = task_for_seed(int(seed))
            answer, method = extract_integer_answer(response)
            recalculated = bool(task.kind == kind and task.question == question
                                and task.expected == expected and answer == expected)
            result.append({'seed': seed, 'kind': kind, 'expected': expected,
                           'response': (response or '')[:300], 'parsed': answer,
                           'parse_method': method, 'correct': recalculated,
                           'legacy_correct': bool(old_correct), 'model': model})
        return result

    def training_examples(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT seed,kind,question,expected,correct FROM attempts WHERE split='learn' ORDER BY id"
        ).fetchall()
        # Deduplicate tasks: independently-computed labels are valid even after an AI failure.
        seen, result = set(), []
        for seed, kind, question, expected, correct in rows:
            if seed in seen:
                continue
            seen.add(seed)
            result.append({"seed": seed, "kind": kind, "question": question,
                           "expected": expected, "label_source": "deterministic_generator", "verified": True})
        return result

    def last_chat(self, n: int = 8) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT role, content FROM chat ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [("User" if role == "user" else "Assistant", str(content)) for role, content in reversed(rows)]

    def save_chat(self, role: str, content: str) -> None:
        self.db.execute("INSERT INTO chat (time,role,content) VALUES (?,?,?)", (stamp(), role, content))
        self.db.commit()

    def close(self):
        self.db.close()


class CloudStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.attempts_path = root / "attempts.jsonl"
        self.examples_path = root / "verified_examples.jsonl"
        self.state_path = root / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {"next_seed": 0}
        # Safety: ensure a seed cannot be inadvertently replayed after an interrupted run.
        recorded = read_jsonl(self.attempts_path)
        if recorded:
            self.state["next_seed"] = max(
                int(self.state.get("next_seed", 0)),
                max(int(row["seed"]) for row in recorded if row.get("split") == "learn") + 1,
            )
        self._examples = read_jsonl(self.examples_path)

    def get_next_seed(self) -> int:
        return int(self.state.get("next_seed", 0))

    def examples(self, kind: str, n: int = 2) -> list[dict]:
        return [x for x in reversed(self._examples) if x.get("kind") == kind][:n]

    def add(self, row: dict) -> None:
        append_jsonl(self.attempts_path, row)
        if row["split"] == "learn":
            sample = {k: row[k] for k in ("seed", "kind", "question", "expected", "label_source")}
            sample["verified"] = True
            append_jsonl(self.examples_path, sample)
            self._examples.append(sample)
            self.state["next_seed"] = max(self.get_next_seed(), int(row["seed"]) + 1)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")
            tmp.replace(self.state_path)

    def stats(self) -> dict:
        rows = [x for x in read_jsonl(self.attempts_path) if x.get('split') == 'learn']
        recalculated = original = rescued = 0
        for row in rows:
            original += int(bool(row.get('correct')))
            t = task_for_seed(int(row['seed']))
            answer, _ = extract_integer_answer(row.get('model_response', ''))
            correct = bool(t.question == row.get('question') and t.kind == row.get('kind')
                           and t.expected == row.get('expected') and answer == t.expected)
            recalculated += int(correct)
            rescued += int(correct and not row.get('correct'))
        return {'total': len(rows), 'correct': recalculated,
                'legacy_correct': original, 'rescued_from_format': rescued,
                'accuracy': round(recalculated / len(rows), 4) if rows else None}

    def write_summary(self) -> None:
        s = self.stats()
        agent_path = self.root / 'agent_state.json'
        agent = json.loads(agent_path.read_text(encoding='utf-8')) if agent_path.exists() else {}
        calls = agent.get('calls', {})
        n_calls = sum(calls.values())
        duration = agent.get('seconds', {})
        n_seconds = sum(duration.values())
        verifier_seconds = duration.get('verifier', 0.0)
        self.root.joinpath("summary.md").write_text(
            "# Online learning experiment\n\n"
            f"Updated (UTC): {stamp()}\n\n"
            f"- Learning tasks attempted: {s['total']}\n"
            f"- Re-scored mathematically correct model responses: {s['correct']}\n"
            f"- Original strict-format successes: {s['legacy_correct']}\n"
            f"- Recovered formatting-only failures: {s['rescued_from_format']}\n"
            f"- Historical task accuracy: {s['accuracy']}\n"
            "- Persistent changes: independently verified examples and attempt logs\n"
            f"- Multi-agent cycles: {agent.get('cycles', 0)}\n"
            f"- Verification inference-call share: {calls.get('verifier', 0)}/{n_calls} "
            f"({(100*calls.get('verifier', 0)/n_calls if n_calls else 0):.1f}% when calls > 0)\n"
            f"- Measured verifier inference wall-time: {verifier_seconds:.1f}/{n_seconds:.1f} seconds "
            f"({(100*verifier_seconds/n_seconds if n_seconds else 0):.1f}%)\n"
            "- A bounded scheduler targets >=50% verifier inference time per cycle. Wall-time is NOT GPU electrical energy.\n"
            "- **Neural weights have NOT been changed by GitHub Actions.**\n"
            "- This is a narrow mathematics smoke test, not a general intelligence benchmark.\n",
            encoding="utf-8",
        )
