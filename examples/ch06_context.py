"""Chapter 6: what reaches the model, and where it sits."""

from __future__ import annotations

from ragsys.configs import CONFIGS
from ragsys.context import edges_first
from ragsys.evals import evaluate, load_questions, summarize
from ragsys.llm import MODELS, gather
from ragsys.pipeline import Pipeline


def support_by_position(pipe: Pipeline) -> None:
    q = "How long are audit logs kept on the Enterprise plan?"
    hits = pipe.index.search(q, 8)
    gold = next(
        h.chunk for h in hits if "admin-guide" in h.chunk.id
    )
    others = [h.chunk for h in hits if h.chunk is not gold][:7]
    for place in (0, 3, 7):
        order = others[:place] + [gold] + others[place:]
        passages = [pipe.passage(c) for c in order]
        row = f"  gold at position {place + 1} of 8:"
        for model in ("small", "large"):
            ev = gather(q, passages, MODELS[model], frozenset())
            mine = max(
                e.score for e in ev if e.chunk_id == gold.id
            )
            row += f"  {model} {mine:.2f}"
        print(row)


def main() -> None:
    base = CONFIGS["reranked"]
    print("support for the best gold sentence, by position:")
    support_by_position(Pipeline(base))
    print(
        f"\n{'top_k':>5} {'order':>6} {'accuracy':>9}"
        f" {'tokens':>7}"
    )
    questions = load_questions()
    for k in (2, 4, 8):
        for order in ("rank", "edges"):
            pipe = Pipeline(base.but(top_k=k, order=order))
            s = summarize(evaluate(pipe, questions))
            print(
                f"{k:5} {order:>6} {s['accuracy']:9.1%} "
                f"{s['tokens_in']:7}"
            )
    pipe = Pipeline(base.but(top_k=8, compress_to=2))
    s = summarize(evaluate(pipe, questions))
    print(
        f"8 compressed  {s['accuracy']:9.1%} {s['tokens_in']:7}"
    )
    print(
        f"\nedges_first(1..6) -> {edges_first(list(range(1, 7)))}"
    )


if __name__ == "__main__":
    main()
