"""Chapter 16: find the failing stage, then climb the ladder."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.agent import MultiHopAgent
from ragsys.configs import BASELINE, LADDER, PRODUCTION
from ragsys.evals import (
    evaluate,
    is_correct,
    load_questions,
    split,
    summarize,
)
from ragsys.pipeline import Pipeline
from ragsys.trace import failure_profile, to_record


def agent_accuracy(questions) -> float:
    agent = MultiHopAgent(Pipeline(PRODUCTION))
    ok = 0
    for q in questions:
        a = agent.run(
            q.question, user_for(q.user, q.tenant)
        ).answer
        ok += is_correct(q, a.text, a.abstained)
    return ok / len(questions)


def main() -> None:
    questions = load_questions()
    dev, test = split(questions, "dev"), split(questions, "test")
    for cfg in (BASELINE, PRODUCTION):
        profile = failure_profile(Pipeline(cfg), questions)
        print(
            f"{cfg.name:10} "
            + "  ".join(
                f"{k} {v}" for k, v in profile.most_common()
            )
        )
    print(
        f"\n{'step':11} {'dev':>6} {'test':>6} {'leaks':>6} "
        f"{'obeyed':>7} {'tokens':>7}"
    )
    for cfg in LADDER:
        pipe = Pipeline(cfg)
        d = summarize(evaluate(pipe, dev))
        t = summarize(evaluate(pipe, test))
        print(
            f"{cfg.name:11} {d['accuracy']:6.1%} "
            f"{t['accuracy']:6.1%} {d['leaks'] + t['leaks']:6} "
            f"{d['injected'] + t['injected']:7}"
            f" {d['tokens_in']:7}"
        )
    large = Pipeline(PRODUCTION.but(model="large"))
    d = summarize(evaluate(large, dev))
    t = summarize(evaluate(large, test))
    print(
        f"{'large model':11} {d['accuracy']:6.1%} "
        f"{t['accuracy']:6.1%}"
    )
    print(
        f"{'+ multi-hop':11} {agent_accuracy(dev):6.1%} "
        f"{agent_accuracy(test):6.1%}"
    )
    pipe = Pipeline(PRODUCTION)
    r = pipe.run(
        "Who is our technical account manager?",
        user_for("customer", "bramble"),
    )
    print()
    for key, value in to_record(r, pipe).items():
        text = textwrap.fill(
            f"{key:9} {value}", 60, subsequent_indent=" " * 12
        )
        print(textwrap.indent(text, "  "))


if __name__ == "__main__":
    main()
