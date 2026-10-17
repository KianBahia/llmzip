"""Textual terminal app (`llmzip tui`). See app.py for the architecture notes."""


def run(path: str = ".", max_tokens: int = 5000, tab: str = "overview") -> None:
    from llmzip.tui.app import run as _run

    _run(path, max_tokens, tab)
