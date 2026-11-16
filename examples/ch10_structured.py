"""Chapter 10: records from a system of record, plus documents."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.configs import PRODUCTION
from ragsys.pipeline import Pipeline
from ragsys.structured import (
    STRUCTURED_QUESTIONS,
    RecordAssistant,
)


def main() -> None:
    pipe = Pipeline(PRODUCTION)
    bot = RecordAssistant(pipe)
    staff = user_for("support")
    docs_ok = tool_ok = 0
    for question, expected in STRUCTURED_QUESTIONS:
        docs = pipe.run(question, staff).answer.text
        reply = bot.ask(question, staff)
        docs_ok += expected.lower() in docs.lower()
        tool_ok += expected.lower() in reply.answer.text.lower()
        print(f"{reply.route:16}{textwrap.shorten(question, 46)}")
        if reply.follow_up:
            print(f"{'':4} then: {reply.follow_up}")
    n = len(STRUCTURED_QUESTIONS)
    print(
        f"\ndocuments only: {docs_ok}/{n}   records + docs: "
        f"{tool_ok}/{n}"
    )
    outsider = user_for("customer", "orchid")
    reply = bot.ask("What plan is Calder Media on?", outsider)
    print(f"Orchid asks about Calder Media: {reply.route}")


if __name__ == "__main__":
    main()
