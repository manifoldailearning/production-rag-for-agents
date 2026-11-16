"""Chapter 15: latency, cost, approximate search and caching."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.ann import IVFIndex
from ragsys.cache import AnswerCache
from ragsys.configs import CONFIGS, PRODUCTION
from ragsys.cost import estimate
from ragsys.evals import load_questions
from ragsys.pipeline import Pipeline
from ragsys.search import VectorIndex

Q = "What is the minimum schedule interval on the Team plan?"
BIG = 2_000_000


def per_request(name, cfg, scanned: float = 1.0) -> None:
    c = estimate(Pipeline(cfg).run(Q), cfg, BIG, scanned)
    s = c.stages
    print(
        f"{name:12} {c.ms:6.0f} {s['vector']:7.0f} "
        f"{s.get('rerank', 0):7.0f} {s['generate']:6.0f} "
        f"{c.dollars:9.4f}"
    )


def ivf_table() -> None:
    pipe = Pipeline(CONFIGS["sections"])
    exact, ivf = (
        VectorIndex(pipe.chunks),
        IVFIndex(pipe.chunks, 6),
    )
    questions = [q for q in load_questions() if q.gold]
    print(f"IVF over {len(pipe.chunks)} chunks in 6 clusters:")
    for probes in (1, 2, 3, 6):
        ivf.comparisons, overlap = 0, 0.0
        for q in questions:
            want = {
                h.chunk.id for h in exact.search(q.question, 4)
            }
            got = ivf.search(q.question, 4, probes)
            overlap += len(want & {h.chunk.id for h in got}) / 4
        n = len(questions)
        print(
            f"  probes {probes}: recall {overlap / n:5.1%}, "
            f"{ivf.comparisons // n} comparisons per query"
        )


def main() -> None:
    print(f"one request, {BIG:,} chunks (ms, except $)")
    print(
        f"{'':12} {'total':>6} {'vector':>7} {'rerank':>7} "
        f"{'gen':>6} {'dollars':>9}"
    )
    per_request("exact", PRODUCTION)
    per_request("IVF 5%", PRODUCTION, 0.05)
    per_request("no rerank", PRODUCTION.but(rerank=False), 0.05)
    per_request(
        "large model", PRODUCTION.but(model="large"), 0.05
    )
    print()
    ivf_table()
    pipe, user = Pipeline(PRODUCTION), user_for("customer")
    first = "What does the Team plan cost per month?"
    other = "What does the Business plan cost per month?"
    for threshold in (0.90, 0.95):
        cache = AnswerCache("semantic", threshold)
        cache.put(first, user, "v1", pipe.run(first, user).answer)
        hit = cache.get(other, user, "v1")
        served = hit.answer.text if hit else "miss: pipeline runs"
        print(
            f"\nsemantic cache at {threshold}, Business question:"
        )
        print(f"  {textwrap.shorten(served, 58)}")


if __name__ == "__main__":
    main()
