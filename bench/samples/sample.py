"""A tiny word-frequency counter used as a compression sample."""

import re
import sys
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words, ignoring punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def top_words(text: str, n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(tokenize(text))
    return counts.most_common(n)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} FILE", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        text = f.read()
    for word, count in top_words(text):
        print(f"{count:6d}  {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
