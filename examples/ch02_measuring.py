"""Chapter 2: grade retrieval and answers separately."""

from __future__ import annotations

from ragsys.access import user_for
from ragsys.configs import BASELINE
from ragsys.evals import (
    evaluate,
    grade,
    load_questions,
    split,
    summarize,
)
from ragsys.pipeline import Pipeline


def main() -> None:
    questions = load_questions()
    dev, test = split(questions, "dev"), split(questions, "test")
    print(
        f"{len(questions)} questions: {len(dev)} dev, "
        f"{len(test)} test"
    )
    pipe = Pipeline(BASELINE)
    q = next(q for q in questions if q.id == "q003")
    r = pipe.run(q.question, user_for(q.user, q.tenant))
    g = grade(q, r, pipe)
    print(f"\n{q.id}: {q.question}")
    print(f"  expected {q.expected}, must not say {q.wrong[:2]}")
    print(
        f"  gold doc retrieved {g.found}, in context "
        f"{g.in_context}, correct {g.correct}"
    )
    print(f"  answer used {r.answer.used}")
    print("\nbaseline on the dev split:")
    for key, value in summarize(evaluate(pipe, dev)).items():
        print(f"  {key:15} {value}")


if __name__ == "__main__":
    main()
