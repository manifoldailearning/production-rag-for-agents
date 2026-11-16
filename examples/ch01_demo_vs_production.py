"""Chapter 1: a demo-quality pipeline against real questions."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.configs import BASELINE
from ragsys.evals import (
    by_type,
    evaluate,
    load_questions,
    summarize,
)
from ragsys.pipeline import Pipeline


def show(pipe: Pipeline, question: str, **who) -> None:
    answer = pipe.run(question, user_for(**who)).answer.text
    print(textwrap.fill(f"Q: {question}", 62))
    short = textwrap.shorten(answer, 160, placeholder=" ...")
    print(textwrap.fill(f"A: {short}", 62) + "\n")


def main() -> None:
    pipe = Pipeline(BASELINE)
    show(
        pipe,
        "What does the Team plan cost per month?",
        name="customer",
    )
    show(
        pipe,
        "Is there a mobile app for Larkspur Cloud?",
        name="customer",
    )
    grades = evaluate(pipe, load_questions())
    s = summarize(grades)
    print(
        f"accuracy {s['accuracy']:.1%} on {len(grades)} questions"
    )
    print(
        f"answers built on documents the user can't read: "
        f"{s['leaks']}"
    )
    print(
        f"answers that obeyed a planted instruction: "
        f"{s['injected']}"
    )
    for kind, (ok, n) in by_type(grades).items():
        print(f"  {kind:13} {ok:2}/{n}")


if __name__ == "__main__":
    main()
