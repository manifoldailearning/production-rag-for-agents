"""Chapter 4: lexical, dense and hybrid retrieval compared."""

from __future__ import annotations

from collections import defaultdict

from ragsys.chunking import chunk_corpus
from ragsys.corpus import load_corpus
from ragsys.evals import load_questions
from ragsys.search import BM25Index, HybridIndex, VectorIndex

TYPES = ("lookup", "paraphrase", "identifier", "temporal")


def hit_rate(index, questions, k: int) -> dict[str, float]:
    """Share of questions with a gold document in the top k."""
    found: dict[str, list[bool]] = defaultdict(list)
    for q in questions:
        docs = {
            h.chunk.doc_id for h in index.search(q.question, k)
        }
        hit = bool(docs & set(q.gold))
        found[q.type].append(hit)
        found["all"].append(hit)
    return {t: sum(v) / len(v) for t, v in found.items()}


def main() -> None:
    chunks = chunk_corpus(load_corpus(), "section")
    questions = [q for q in load_questions() if q.gold]
    indexes = {
        "bm25": BM25Index(chunks),
        "dense": VectorIndex(chunks),
        "hybrid": HybridIndex(chunks),
    }
    for k in (1, 2):
        print(
            f"hit@{k}      all"
            + "".join(f"{t:>11}" for t in TYPES)
        )
        for name, index in indexes.items():
            r = hit_rate(index, questions, k)
            cells = "".join(f"{r[t]:11.0%}" for t in TYPES)
            print(f"{name:8} {r['all']:5.0%}{cells}")
        print()
    for q in (
        "What does error LK-4012 mean?",
        "I want to stop paying. How do I end my plan?",
    ):
        print(q)
        for name, index in indexes.items():
            top = index.search(q, 1)[0].chunk.doc_id
            print(f"  {name:7} top hit: {top}")


if __name__ == "__main__":
    main()
