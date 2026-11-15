"""SQLite-backed local history; JSONL-backed online history.

A trusted label is computed from the task generator, never from an AI's
self-assessment. Records never contain local user chats in the cloud build.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import sqlite3


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
        rows = self.db.execute(
            "SELECT question, expected FROM attempts WHERE correct=1 "
            "AND split='learn' AND kind=? ORDER BY id DESC LIMIT ?", (kind, n),
        ).fetchall()
        return [{"question": row[0], "expected": row[1]} for row in rows]

    def stats(self) -> dict:
        total, correct = self.db.execute(
            "SELECT COUNT(*),COALESCE(SUM(correct),0) FROM attempts WHERE split='learn'"
        ).fetchone()
        return {"total": total, "correct": correct, "accuracy": round(correct / total, 4) if total else None}

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
        rows = [x for x in read_jsonl(self.attempts_path) if x.get("split") == "learn"]
        correct = sum(bool(x.get("correct")) for x in rows)
        return {"total": len(rows), "correct": correct,
                "accuracy": round(correct / len(rows), 4) if rows else None}

    def write_summary(self) -> None:
        s = self.stats()
        agent_path = self.root / 'agent_state.json'
        agent = json.loads(agent_path.read_text(encoding='utf-8')) if agent_path.exists() else {}
        calls = agent.get('calls', {})
        n_calls = sum(calls.values())
        self.root.joinpath("summary.md").write_text(
            "# Online learning experiment\n\n"
            f"Updated (UTC): {stamp()}\n\n"
            f"- Learning tasks attempted: {s['total']}\n"
            f"- Independently correct initial answers: {s['correct']}\n"
            f"- Historical task accuracy: {s['accuracy']}\n"
            "- Persistent changes: independently verified examples and attempt logs\n"
            f"- Multi-agent cycles: {agent.get('cycles', 0)}\n"
            f"- Verification inference-call share: {calls.get('verifier', 0)}/{n_calls} "
            f"({(100*calls.get('verifier', 0)/n_calls if n_calls else 0):.1f}% when calls > 0)\n"
            "- Budget refers to inference CALLS, not measured GPU energy or elapsed compute.\n"
            "- **Neural weights have NOT been changed by GitHub Actions.**\n"
            "- This is a narrow mathematics smoke test, not a general intelligence benchmark.\n",
            encoding="utf-8",
        )
