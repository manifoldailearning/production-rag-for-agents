"""Chapter 3: the same documents, chunked three ways."""

from __future__ import annotations

import textwrap

from ragsys.configs import BASELINE, PRODUCTION
from ragsys.context import cost
from ragsys.evals import (
    by_type,
    evaluate,
    load_questions,
    split,
    summarize,
)
from ragsys.pipeline import Pipeline

STRATEGIES = (("fixed", 120), ("sentence", 120), ("section", 160))
TYPES = ("temporal", "long", "lookup")


def main() -> None:
    questions = load_questions()
    print(
        f"{'strategy':9} {'chunks':>6} {'tokens':>6} "
        f"{'acc':>6}  " + "  ".join(f"{t:>8}" for t in TYPES)
    )
    for strategy, size in STRATEGIES:
        cfg = BASELINE.but(chunker=strategy, chunk_size=size)
        pipe = Pipeline(cfg)
        avg = sum(cost(c) for c in pipe.chunks) / len(pipe.chunks)
        grades = evaluate(pipe, questions)
        acc = summarize(grades)["accuracy"]
        kinds = by_type(grades)
        cells = "  ".join(
            f"{kinds[t][0]:>4}/{kinds[t][1]:<3}" for t in TYPES
        )
        print(
            f"{strategy:9} {len(pipe.chunks):6} {avg:6.0f} "
            f"{acc:6.1%}  {cells}"
        )
    print("\nsection chunk size: baseline vs production settings")
    dev = split(questions, "dev")
    for size in (30, 60, 160):
        row = f"  {size:3} words"
        for cfg in (BASELINE.but(chunker="section"), PRODUCTION):
            pipe = Pipeline(cfg.but(chunk_size=size))
            s = summarize(evaluate(pipe, dev))
            row += (
                f"   {s['accuracy']:6.1%} {s['tokens_in']:4} tok"
            )
        print(row)
    section = Pipeline(BASELINE.but(chunker="section"))
    row = next(
        c for c in section.chunks if c.id == "pricing-v3#0"
    )
    print("\nA table row as a section chunk sees it:")
    print(" ", row.header)
    line = row.text.splitlines()[5]
    print(" ", textwrap.shorten(line, 58, placeholder=" ..."))


if __name__ == "__main__":
    main()
