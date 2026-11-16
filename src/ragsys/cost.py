"""Latency and cost per request, stage by stage.

Each stage of the pipeline has a price in milliseconds and some
have a price in dollars. The numbers here are planning
assumptions, stated in one place so they can be replaced with
measurements from your own stack. What matters is the shape:
the model call dominates, reranking scales with candidates and
exact vector search scales with the corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsys.llm import MODELS
from ragsys.pipeline import RagConfig, Result

ASSUMED_MS = {
    "lexical_per_1k_chunks": 1.0,
    "embed_query": 25.0,
    "vector_per_1k_chunks": 2.0,
    "rerank_per_pair": 8.0,
    "prefill_per_token": 0.05,
}


@dataclass(frozen=True)
class Cost:
    stages: dict[str, float]
    dollars: float

    @property
    def ms(self) -> float:
        return round(sum(self.stages.values()), 1)


def estimate(
    result: Result, config: RagConfig, corpus_chunks: int
) -> Cost:
    """Latency per stage and dollars for one answered request."""
    a, model = ASSUMED_MS, MODELS[config.model]
    k = corpus_chunks / 1000
    stages: dict[str, float] = {}
    if config.retriever in ("bm25", "hybrid"):
        stages["lexical"] = 2.0 + a["lexical_per_1k_chunks"] * k
    if config.retriever in ("dense", "hybrid"):
        stages["embed"] = a["embed_query"]
        stages["vector"] = a["vector_per_1k_chunks"] * k
    if config.rerank:
        stages["rerank"] = a["rerank_per_pair"] * len(result.hits)
    tin, tout = result.answer.tokens_in, result.answer.tokens_out
    stages["prefill"] = a["prefill_per_token"] * tin
    stages["generate"] = model.ms_per_token * tout
    dollars = (tin + tout) / 1000 * model.cost_per_1k
    rounded = {s: round(v, 1) for s, v in stages.items()}
    return Cost(rounded, round(dollars, 6))
