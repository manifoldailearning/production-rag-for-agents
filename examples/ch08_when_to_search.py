"""Chapter 8: should the agent search at all?"""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.agent import GENERAL_QUESTIONS, Assistant, closed_book
from ragsys.configs import PRODUCTION
from ragsys.evals import is_correct, load_questions
from ragsys.pipeline import Pipeline

SKIP = {"unanswerable", "restricted", "multihop"}


def main() -> None:
    q = "Within how many days can annual plans be refunded?"
    print(
        textwrap.fill(f"closed book: {closed_book(q).text}", 62)
    )
    print()
    company = [x for x in load_questions() if x.type not in SKIP]
    pipe = Pipeline(PRODUCTION)
    print(
        f"{'policy':7} {'company':>9} {'general':>8} "
        f"{'searches':>9} {'tokens':>7}"
    )
    for policy in ("never", "always", "router"):
        bot = Assistant(pipe, policy)
        ok_c = ok_g = searches = tokens = 0
        for c in company:
            r = bot.ask(c.question, user_for(c.user, c.tenant))
            a = r.answer
            ok_c += is_correct(c, a.text, a.abstained)
            searches += r.searched
            tokens += a.tokens_in
        for question, expected in GENERAL_QUESTIONS:
            r = bot.ask(question)
            ok_g += expected.lower() in r.answer.text.lower()
            searches += r.searched
            tokens += r.answer.tokens_in
        n = len(company) + len(GENERAL_QUESTIONS)
        print(
            f"{policy:7} {ok_c:5}/{len(company)} {ok_g:5}/"
            f"{len(GENERAL_QUESTIONS)} {searches:9} "
            f"{tokens // n:7}"
        )


if __name__ == "__main__":
    main()
