"""A simulated cross-encoder reranker.

First-stage search is fast but coarse: it compares a query with
every chunk independently. A reranker reads the query and one
candidate together and rescores a short list, which is slower
per item but far more precise. Here the joint reading is the
same relevance scorer the simulated model uses.
"""

from __future__ import annotations

from ragsys.relevance import answer_fit, relevance
from ragsys.search import Hit, top
from ragsys.text import sentences

MS_PER_PAIR = 8.0


def score_chunk(question: str, header: str, text: str) -> float:
    """Best sentence, read with the chunk's header for context.

    Like a cross-encoder, it checks that the passage names what
    the question names and has the shape of the answer asked for.
    """
    lines = sentences(text) or [text]
    best = max(
        relevance(
            question, f"{header}. {s}" if header else s, 0.4
        )
        * answer_fit(question, s)
        for s in lines
    )
    return round(best, 4)


def rerank(question: str, hits: list[Hit], k: int) -> list[Hit]:
    """Rescore candidates jointly with the question; keep k."""
    rescored = [
        Hit(
            h.chunk,
            score_chunk(question, h.chunk.header, h.chunk.text),
        )
        for h in hits
    ]
    return top(rescored, k)
