"""Answer from records and documents together.

Some facts live in systems of record, not in documents: which
plan a customer is on, how many pipelines it uses, when it
renews. A document index cannot answer "what is Calder Media's
log retention?", because no document mentions Calder Media. The
assistant looks the account up with a tool, then asks the
knowledge base about the plan the record names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragsys.access import User, user_for
from ragsys.llm import ABSTAIN, Answer
from ragsys.pipeline import Pipeline


@dataclass(frozen=True)
class Account:
    name: str
    tenant: str
    plan: str
    region: str
    pipelines: int
    renews: str


ACCOUNTS = (
    Account(
        "Orchid Retail",
        "orchid",
        "Business",
        "us-east",
        64,
        "1 April 2027",
    ),
    Account(
        "Bramble Logistics",
        "bramble",
        "Enterprise",
        "eu-west",
        210,
        "1 July 2027",
    ),
    Account(
        "Fennel Health",
        "fennel",
        "Team",
        "us-east",
        18,
        "15 January 2027",
    ),
    Account(
        "Juniper Foods",
        "juniper",
        "Starter",
        "us-east",
        4,
        "9 December 2026",
    ),
    Account(
        "Calder Media",
        "calder",
        "Business",
        "ap-south",
        41,
        "1 February 2027",
    ),
)


def find_account(question: str) -> Account | None:
    """Exact name match only: the tool never guesses."""
    for a in ACCOUNTS:
        if a.name.lower() in question.lower():
            return a
    return None


def may_see(user: User, account: Account) -> bool:
    """Staff see every account; customers only their own."""
    return "staff" in user.groups or user.tenant == account.tenant


def describe(a: Account) -> str:
    return (
        f"{a.name} is on the {a.plan} plan in {a.region}, uses "
        f"{a.pipelines} pipelines and renews on {a.renews}."
    )


PLAN_FACTS = re.compile(
    r"retention|logs?|support|response|price|cost|sso|uptime"
    r"|schedule|interval|audit|sign-on",
    re.I,
)


@dataclass
class Reply:
    route: str  # docs, record, record+docs or denied
    answer: Answer
    follow_up: str = ""


class RecordAssistant:
    """Route account questions to a lookup, then to the docs."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def ask(
        self, question: str, user: User | None = None
    ) -> Reply:
        user = user or user_for("support")
        account = find_account(question)
        if account is None:
            result = self.pipeline.run(question, user)
            return Reply("docs", result.answer)
        if not may_see(user, account):
            return Reply(
                "denied", Answer(ABSTAIN, abstained=True)
            )
        record = describe(account)
        source = (f"record:{account.tenant}",)
        if not PLAN_FACTS.search(question):
            return Reply("record", Answer(record, source))
        follow = re.sub(
            rf"{account.name}(?:'s plan|'s)?",
            f"the {account.plan} plan",
            question,
        )
        docs = self.pipeline.run(follow, user).answer
        merged = Answer(
            f"{record} {docs.text}",
            source + docs.used,
            docs.citations,
            docs.abstained,
            docs.injected,
            docs.tokens_in,
            docs.tokens_out,
        )
        return Reply("record+docs", merged, follow)


STRUCTURED_QUESTIONS = (
    ("What plan is Calder Media on?", "Business"),
    ("When does Fennel Health renew?", "15 January 2027"),
    ("Which region does Calder Media run in?", "ap-south"),
    ("How many pipelines does Juniper Foods use?", "4 pipelines"),
    (
        "What log retention applies to Calder Media's plan?",
        "90 days",
    ),
    ("How long are logs kept for Fennel Health?", "30 days"),
    (
        "What response target does Bramble Logistics get?",
        "1-hour",
    ),
    ("Does Juniper Foods get single sign-on?", "not available"),
)
