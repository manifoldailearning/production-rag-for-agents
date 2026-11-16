"""Chapter 9: questions that need a second search."""

from __future__ import annotations

import textwrap

from ragsys.agent import MultiHopAgent
from ragsys.configs import PRODUCTION
from ragsys.evals import is_correct, load_questions
from ragsys.pipeline import Pipeline


def main() -> None:
    pipe = Pipeline(PRODUCTION)
    agent = MultiHopAgent(pipe)
    chained = [
        q for q in load_questions() if q.type == "multihop"
    ]
    single = multi = t_single = t_multi = 0
    for q in chained:
        one = pipe.run(q.question).answer
        hops = agent.run(q.question).answer
        single += is_correct(q, one.text, one.abstained)
        multi += is_correct(q, hops.text, hops.abstained)
        t_single += one.tokens_in
        t_multi += hops.tokens_in
    n = len(chained)
    print(
        f"single search: {single}/{n} correct, "
        f"{t_single // n} tokens in"
    )
    print(
        f"multi-hop:     {multi}/{n} correct, "
        f"{t_multi // n} tokens in\n"
    )
    result = agent.run(chained[2].question)
    print(textwrap.fill(result.question, 62))
    for i, hop in enumerate(result.hops, 1):
        print(f"  hop {i}: {hop.query}")
        for line in textwrap.wrap(hop.answer.text, 54):
            print(f"         {line}")


if __name__ == "__main__":
    main()
