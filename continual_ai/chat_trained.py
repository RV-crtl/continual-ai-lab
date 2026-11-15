"""Explicit chat route for a promoted adapter; base Ollama chat remains unchanged."""
from __future__ import annotations
from pathlib import Path
from .hf_client import HuggingFaceClient


def main():
    root = Path('data/local/training')
    client = HuggingFaceClient.active(root)
    print('Loaded promoted adapter:', client.model)
    print('NOTE: Promoted adapter was evaluated on narrow arithmetic tasks only.')
    print('Type /exit to stop.')
    while True:
        try:
            message = input('You> ').strip()
        except EOFError:
            break
        if message.lower() in {'/exit', '/quit'}:
            break
        if message:
            print('AI>', client.generate(message, max_tokens=384), flush=True)


if __name__ == '__main__':
    main()
