"""Lexical, dense and hybrid search over chunks.

BM25 rewards exact words and rare terms such as error codes.
Dense search rewards meaning, so paraphrases match. Hybrid
search fuses the two rankings with reciprocal rank fusion,
which needs no score calibration between the two systems.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from ragsys.chunking import Chunk
from ragsys.embed import cosine, embed
from ragsys.text import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 with the usual k1 and b parameters."""

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [
            Counter(tokenize(c.indexed_text)) for c in chunks
        ]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.lengths) / max(len(chunks), 1)
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {
            t: math.log(1 + (n - d + 0.5) / (d + 0.5))
            for t, d in df.items()
        }

    def score(self, terms: list[str], i: int) -> float:
        tf, length = self.tf[i], self.lengths[i]
        norm = self.k1 * (1 - self.b + self.b * length / self.avg)
        return sum(
            self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            for t in terms
            if t in tf
        )

    def search(
        self, query: str, k: int = 10, allowed: set | None = None
    ) -> list[Hit]:
        terms = tokenize(query)
        scored = [
            Hit(c, self.score(terms, i))
            for i, c in enumerate(self.chunks)
            if allowed is None or c.doc_id in allowed
        ]
        return top([h for h in scored if h.score > 0], k)


class VectorIndex:
    """Exact nearest-neighbour search over chunk embeddings."""

    def __init__(self, chunks: list[Chunk], cache=None):
        self.chunks = chunks
        cache = {} if cache is None else cache
        self.vectors = [
            cache.get(c.id) or embed(c.indexed_text)
            for c in chunks
        ]
        self.comparisons = 0

    def search(
        self, query: str, k: int = 10, allowed: set | None = None
    ) -> list[Hit]:
        q = embed(query)
        scored = [
            Hit(c, cosine(q, v))
            for c, v in zip(
                self.chunks, self.vectors, strict=True
            )
            if allowed is None or c.doc_id in allowed
        ]
        self.comparisons += len(scored)
        return top([h for h in scored if h.score > 0], k)


def top(hits: list[Hit], k: int) -> list[Hit]:
    """Best k hits; ties broken by chunk id for stable output."""
    return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:k]


def rrf(rankings: list[list[Hit]], k: int = 10, c: int = 60):
    """Reciprocal rank fusion of several ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.id] += 1 / (c + rank)
            chunks[hit.chunk.id] = hit.chunk
    fused = [Hit(chunks[i], s) for i, s in scores.items()]
    return top(fused, k)


class HybridIndex:
    """BM25 and dense search fused with RRF."""

    def __init__(self, chunks: list[Chunk], cache=None):
        self.chunks = chunks
        self.lexical = BM25Index(chunks)
        self.dense = VectorIndex(chunks, cache)

    def search(
        self, query: str, k: int = 10, allowed: set | None = None
    ) -> list[Hit]:
        pool = max(k, 20)
        return rrf(
            [
                self.lexical.search(query, pool, allowed),
                self.dense.search(query, pool, allowed),
            ],
            k,
        )


def build_index(chunks: list[Chunk], kind: str = "hybrid"):
    """Factory used by the pipeline configuration."""
    return {
        "bm25": BM25Index,
        "dense": VectorIndex,
        "hybrid": HybridIndex,
    }[kind](chunks)
