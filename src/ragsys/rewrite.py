"""Rewrite the user's question before it reaches the index.

Users describe problems in their own words; documents use the
product's words. Expansion appends the canonical term for each
concept the question mentions, so "money back" also searches
for "refund". The original question is kept for the model.
"""

from __future__ import annotations

from ragsys.embed import CONCEPTS, concepts

SKIP = {"duration", "price", "plan", "support", "error"}


def expand(question: str) -> str:
    """Question plus canonical words for the concepts it uses."""
    lowered = question.lower()
    extra = [
        CONCEPTS[c][0]
        for c in concepts(question)
        if c not in SKIP and CONCEPTS[c][0] not in lowered
    ]
    return f"{question} {' '.join(extra)}".strip()
