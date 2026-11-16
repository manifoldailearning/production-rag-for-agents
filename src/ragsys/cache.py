"""Answer caches, and the two ways they go wrong.

A cache returns a stored answer instead of running the pipeline.
An exact cache matches normalized question text. A semantic
cache matches questions whose embeddings are close, which raises
the hit rate and also the chance of returning the answer to a
different question. Both must key on who is asking and on the
index version, or they leak and serve stale answers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragsys.access import User
from ragsys.embed import Vector, cosine, embed
from ragsys.llm import Answer


def normalize(question: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", question.lower()).strip()


def scope(user: User, version: str) -> str:
    """Who may share a cached answer, and for which index."""
    return (
        f"{','.join(sorted(user.groups))}|{user.tenant}|{version}"
    )


@dataclass
class Entry:
    question: str
    vector: Vector
    scope: str
    answer: Answer


@dataclass
class AnswerCache:
    mode: str = "exact"  # or "semantic"
    threshold: float = 0.95
    scoped: bool = True
    entries: list[Entry] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def get(self, question: str, user: User, version: str):
        key = scope(user, version) if self.scoped else ""
        vec = embed(question)
        for e in self.entries:
            if e.scope != key:
                continue
            same = normalize(e.question) == normalize(question)
            close = cosine(vec, e.vector) >= self.threshold
            if same or (self.mode == "semantic" and close):
                self.hits += 1
                return e
        self.misses += 1
        return None

    def put(self, question, user, version, answer) -> None:
        key = scope(user, version) if self.scoped else ""
        self.entries.append(
            Entry(question, embed(question), key, answer)
        )
