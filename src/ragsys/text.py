"""Text utilities shared by every stage of the pipeline.

Retrieval quality depends on small, boring decisions: how text
is split into words, which words are ignored and how sentences
are found. Keeping them in one module means the index and the
query side always agree.
"""

from __future__ import annotations

import math
import re

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how
    i if in is it its my of on or our so that the their them then
    there these this to was we what when where which who will
    with you your yes no not after before than per each every
    all any into about up out also only same more most""".split()
)

SUFFIXES = (
    ("ations", 4),
    ("ation", 4),
    ("ing", 3),
    ("ies", 3),
    ("ed", 3),
    ("s", 3),
)
_WORD = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9|])")


def stem(word: str) -> str:
    """A tiny suffix stripper; enough to match common forms."""
    if any(ch.isdigit() for ch in word) or len(word) <= 3:
        return word
    for suffix, keep in SUFFIXES:
        if (
            word.endswith(suffix)
            and len(word) - len(suffix) >= keep
        ):
            if suffix == "s" and word.endswith("ss"):
                break
            root = word[: -len(suffix)]
            if suffix == "ies":
                root += "y"
            elif suffix != "s" and root[-1] == root[-2] != "s":
                root = root[:-1]  # cancelled -> cancel
            word = root
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]  # release, released -> releas
    return word


def words(text: str) -> list[str]:
    """Lowercased word tokens, keeping codes like lk-4012."""
    return _WORD.findall(text.lower())


def tokenize(text: str) -> list[str]:
    """Stemmed content words: the unit of lexical matching."""
    return [stem(w) for w in words(text) if w not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Split prose into sentences and table rows into lines."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if not line or set(line) <= set("|- "):
            continue
        out.extend(
            s.strip() for s in _SENT.split(line) if s.strip()
        )
    return out


def count_tokens(text: str) -> int:
    """Approximate model tokens: about 1.3 per English word."""
    return math.ceil(len(text.split()) * 1.3)
