"""Approximate nearest-neighbour search with an inverted file.

Exact vector search compares the query with every chunk, which
is fine for thousands of chunks and too slow for hundreds of
millions. An inverted file (IVF) groups vectors into clusters
around centroids and searches only the closest few clusters.
Fewer comparisons, at the price of sometimes missing a match.
"""

from __future__ import annotations

import math

from ragsys.chunking import Chunk
from ragsys.embed import Vector, cosine, embed
from ragsys.search import Hit, top


def mean(vectors: list[Vector]) -> Vector:
    total: dict[str, float] = {}
    for v in vectors:
        for k, x in v.items():
            total[k] = total.get(k, 0.0) + x
    norm = math.sqrt(sum(x * x for x in total.values())) or 1.0
    return {k: x / norm for k, x in total.items()}


class IVFIndex:
    """Cluster vectors; search the nearest `probes` clusters."""

    def __init__(
        self, chunks: list[Chunk], lists: int = 6, rounds: int = 5
    ):
        self.chunks = chunks
        self.vectors = [embed(c.indexed_text) for c in chunks]
        step = max(len(chunks) // lists, 1)
        self.centroids = self.vectors[::step][:lists]
        for _ in range(rounds):
            groups = self.assign()
            self.centroids = [
                mean([self.vectors[i] for i in g]) if g else c
                for g, c in zip(
                    groups, self.centroids, strict=True
                )
            ]
        self.lists = self.assign()
        self.comparisons = 0

    def nearest(self, v: Vector) -> int:
        scores = [cosine(v, c) for c in self.centroids]
        return max(range(len(scores)), key=scores.__getitem__)

    def assign(self) -> list[list[int]]:
        groups: list[list[int]] = [[] for _ in self.centroids]
        for i, v in enumerate(self.vectors):
            groups[self.nearest(v)].append(i)
        return groups

    def search(self, query: str, k: int = 10, probes: int = 1):
        q = embed(query)
        order = sorted(
            range(len(self.centroids)),
            key=lambda j: -cosine(q, self.centroids[j]),
        )
        self.comparisons += len(self.centroids)
        hits = []
        for j in order[:probes]:
            for i in self.lists[j]:
                self.comparisons += 1
                hits.append(
                    Hit(
                        self.chunks[i], cosine(q, self.vectors[i])
                    )
                )
        return top(hits, k)
