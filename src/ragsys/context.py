"""Assemble the context window from ranked chunks.

Retrieval returns a ranking; the model needs a context. Between
the two sit decisions that change answers: how many chunks fit
the token budget, whether near-duplicates waste space, where the
strongest evidence sits, and whether long chunks are trimmed to
the sentences that matter.
"""

from __future__ import annotations

from dataclasses import replace

from ragsys.chunking import Chunk
from ragsys.relevance import relevance
from ragsys.text import count_tokens, sentences, tokenize


def cost(chunk: Chunk) -> int:
    return count_tokens(chunk.indexed_text)


def fit_budget(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Keep chunks in rank order while they fit the budget."""
    kept, used = [], 0
    for c in chunks:
        if used + cost(c) <= budget:
            kept.append(c)
            used += cost(c)
    return kept


def similar(a: Chunk, b: Chunk, threshold: float = 0.8) -> bool:
    ta, tb = set(tokenize(a.text)), set(tokenize(b.text))
    union = ta | tb
    return bool(union) and len(ta & tb) / len(union) >= threshold


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks that repeat an earlier, higher-ranked chunk."""
    kept: list[Chunk] = []
    for c in chunks:
        if not any(similar(c, k) for k in kept):
            kept.append(c)
    return kept


def edges_first(chunks: list[Chunk]) -> list[Chunk]:
    """Put the best chunks at the start and end, weakest mid."""
    front, back = chunks[0::2], chunks[1::2]
    return front + back[::-1]


def compress(question: str, chunk: Chunk, keep: int = 2) -> Chunk:
    """Keep only the sentences most relevant to the question."""
    lines = sentences(chunk.text)
    if len(lines) <= keep:
        return chunk
    ranked = sorted(
        range(len(lines)),
        key=lambda i: -relevance(question, lines[i]),
    )
    chosen = sorted(ranked[:keep])
    return replace(
        chunk, text="\n".join(lines[i] for i in chosen)
    )


def assemble(
    question: str,
    ranked: list[Chunk],
    top_k: int = 4,
    budget: int = 1500,
    order: str = "rank",
    remove_duplicates: bool = False,
    compress_to: int = 0,
) -> list[Chunk]:
    """From a ranking to the passages the model will read."""
    chunks = dedupe(ranked) if remove_duplicates else list(ranked)
    if compress_to:
        chunks = [
            compress(question, c, compress_to) for c in chunks
        ]
    chunks = fit_budget(chunks[:top_k], budget)
    return edges_first(chunks) if order == "edges" else chunks
