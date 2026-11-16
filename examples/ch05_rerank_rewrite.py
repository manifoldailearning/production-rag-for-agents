"""Chapter 5: query expansion and reranking."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.configs import CONFIGS
from ragsys.evals import evaluate, load_questions, summarize
from ragsys.pipeline import Pipeline
from ragsys.rewrite import expand


def ranks(pipe: Pipeline, questions):
    """Top-2 hits and mean reciprocal rank of gold docs."""
    top2 = mrr = 0.0
    for q in questions:
        r = pipe.run(q.question, user_for(q.user, q.tenant))
        ids = [h.chunk.doc_id for h in r.hits]
        top2 += bool(set(q.gold) & set(ids[:2]))
        first = next(
            (i for i, d in enumerate(ids, 1) if d in q.gold), None
        )
        mrr += 1 / first if first else 0
    return top2 / len(questions), mrr / len(questions)


def first_gold(pipe: Pipeline, q) -> int | None:
    r = pipe.run(q.question, user_for(q.user, q.tenant))
    ids = [h.chunk.doc_id for h in r.hits]
    return next(
        (i for i, d in enumerate(ids, 1) if d in q.gold), None
    )


def main() -> None:
    q = "Can I get my money back if I paid for a year up front?"
    print(textwrap.fill(f"expanded: {expand(q)}", 62) + "\n")
    questions = [q for q in load_questions() if q.gold]
    paraphrase = [q for q in questions if q.type == "paraphrase"]
    sections = CONFIGS["sections"].but(retriever="hybrid")
    steps = {
        "hybrid": sections,
        "+ expansion": CONFIGS["hybrid"],
        "+ reranker": CONFIGS["reranked"],
    }
    print(
        f"{'':12} {'top2':>6} {'MRR':>6} {'para top2':>10} "
        f"{'answers':>8}"
    )
    for name, cfg in steps.items():
        pipe = Pipeline(cfg)
        top2, mrr = ranks(pipe, questions)
        para, _ = ranks(pipe, paraphrase)
        acc = summarize(evaluate(pipe))["accuracy"]
        print(
            f"{name:12} {top2:6.1%} {mrr:6.3f} {para:10.0%} "
            f"{acc:8.1%}"
        )
    print("\nrank of the first gold document, before -> after:")
    before = Pipeline(CONFIGS["hybrid"])
    after = Pipeline(CONFIGS["reranked"])
    for q in questions:
        if q.id in ("q003", "q086", "q032"):
            text = textwrap.shorten(q.question, 44)
            was, now = first_gold(before, q), first_gold(after, q)
            print(f"  {q.id} {text:45} {was} -> {now}")
    k2 = Pipeline(CONFIGS["reranked"].but(top_k=2))
    s = summarize(evaluate(k2))
    print(
        f"\nreranked, top 2 only: answers {s['accuracy']:.1%}, "
        f"{s['tokens_in']} tokens in"
    )


if __name__ == "__main__":
    main()
