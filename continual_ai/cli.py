"""Command-line interface for the local and hosted prototype."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from .agent import learning_round, evaluate
from .hardware import detect
from .ollama_client import OllamaClient
from .storage import CloudStore, LocalStore, append_jsonl
from .dashboard import serve

DEFAULT_DB = Path(os.getenv("CAI_DB", "data/local/learning.sqlite3"))


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="continual-ai", description="Local AI with independently verified learning experiments")
    command = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "chat", "learn", "loop", "benchmark", "export", "dashboard", "cloud"]:
        cmd = command.add_parser(name)
        if name in ("doctor", "chat", "learn", "loop", "benchmark", "cloud"):
            cmd.add_argument("--model", default=os.getenv("CAI_MODEL", "qwen3:0.6b" if name == "cloud" else "qwen3:4b"))
            cmd.add_argument("--endpoint", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        if name in ("learn", "cloud"):
            cmd.add_argument("--rounds", type=int, default=5 if name == "learn" else 8)
        if name == "cloud":
            cmd.add_argument("--state-dir", type=Path, default=Path("data/cloud"))
            cmd.add_argument("--max-seconds", type=int, default=1500)
        if name == "loop":
            cmd.add_argument("--pause", type=float, default=2.0)
            cmd.add_argument("--max-rounds", type=int, default=0, help="0 = until manually stopped")
        if name == "benchmark":
            cmd.add_argument("--n", type=int, default=12)
        if name == "export":
            cmd.add_argument("--out", type=Path, default=Path("data/local/verified_sft.jsonl"))
        if name == "dashboard":
            cmd.add_argument("--port", type=int, default=8765)
        if name in ("doctor", "chat", "learn", "loop", "benchmark", "export", "dashboard"):
            cmd.add_argument("--db", type=Path, default=DEFAULT_DB)
    return parser


def print_round(row: dict, counter: int):
    status = "PASS" if row["correct"] else "FAIL"
    print(f"{counter:04d} [{status}] {row['kind']} seed={row['seed']} "
          f"expected={row['expected']} | model={row['model']} | answer={row['model_response'].strip()[:80]}", flush=True)


def main(argv=None) -> int:
    p = args_parser()
    args = p.parse_args(argv)
    try:
        if args.command == "dashboard":
            serve(args.db, args.port)
            return 0
        if args.command == "export":
            store = LocalStore(args.db)
            try:
                rows = store.training_examples()
                args.out.parent.mkdir(parents=True, exist_ok=True)
                with args.out.open("w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"Exported {len(rows)} independently labelled examples to {args.out}")
            finally:
                store.close()
            return 0
        client = OllamaClient(args.model, args.endpoint)
        if args.command == "doctor":
            print("Hardware:", json.dumps(detect(), indent=2))
            print("Ollama:")
            print("  installed models:", client.models())
            print("  chosen model:", args.model)
            if args.model not in client.models():
                print(f"  NOTICE: pull model with: ollama pull {args.model}")
            return 0
        if args.command == "chat":
            print(f"Local AI chat ({args.model}). Enter /exit to quit. This is not model training.")
            store = LocalStore(args.db)
            history: list[tuple[str, str]] = store.last_chat(8)
            try:
                while True:
                    try:
                        user = input("\nYou> ").strip()
                    except EOFError:
                        break
                    if user.lower() in {"/exit", "/quit"}:
                        break
                    if not user:
                        continue
                    context = "You are a helpful local AI. Be accurate, and say when unsure.\n"
                    for role, turn in history[-8:]:
                        context += f"{role}: {turn}\n"
                    context += f"User: {user}\nAssistant:"
                    response = client.generate(context, temperature=0.4, max_tokens=700)
                    print("AI>", response)
                    history.extend([("User", user), ("Assistant", response)])
                    store.save_chat("user", user)
                    store.save_chat("assistant", response)
            finally:
                store.close()
            return 0
        if args.command == "benchmark":
            result = evaluate(client, n=args.n)
            print(json.dumps(result, indent=2))
            append_jsonl(Path("data/local/benchmarks.jsonl"), result)
            return 0
        if args.command == "cloud":
            if args.rounds < 1 or args.rounds > 200:
                p.error("cloud rounds must be between 1 and 200")
            store = CloudStore(args.state_dir)
            started = time.monotonic()
            for i in range(args.rounds):
                if time.monotonic() - started > args.max_seconds:
                    print("Time budget reached; completed work is saved.")
                    break
                result = learning_round(store, client)
                print_round(result, i + 1)
            store.write_summary()
            print("Online stats:", store.stats())
            return 0
        store = LocalStore(args.db)
        try:
            if args.command == "learn":
                if args.rounds < 1 or args.rounds > 10000:
                    p.error("rounds must be between 1 and 10000")
                for i in range(args.rounds):
                    print_round(learning_round(store, client), i + 1)
                print("Summary:", store.stats())
            elif args.command == "loop":
                if args.pause < 0 or args.max_rounds < 0:
                    p.error("pause and max-rounds must be nonnegative")
                print("Autonomous verified task loop: press Ctrl+C to stop.")
                i = 0
                while not args.max_rounds or i < args.max_rounds:
                    i += 1
                    print_round(learning_round(store, client), i)
                    if i % 10 == 0:
                        print("Summary:", store.stats(), flush=True)
                    if args.pause:
                        time.sleep(args.pause)
        finally:
            store.close()
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user. Completed learning records remain saved.")
        return 0
    except (RuntimeError, OSError, ValueError) as ex:
        print("ERROR:", ex, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
