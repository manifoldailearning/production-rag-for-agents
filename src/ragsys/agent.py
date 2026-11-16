"""Retrieval as a decision: when to search, and how many times.

An agent should not search for everything. General questions
are answered from the model's own knowledge, while company facts
must come from the knowledge base, because what the model
learned in training is out of date. Questions that chain two
facts need a second search whose query depends on the first
answer. This module covers both decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragsys.access import User, user_for
from ragsys.corpus import load_corpus
from ragsys.llm import GUESS, MODELS, Answer, Passage, generate
from ragsys.pipeline import Pipeline
from ragsys.text import count_tokens

# What the model "knows" without retrieval: general facts, and
# company facts as they stood when its training data was frozen.
GENERAL_FACTS = (
    "SSO stands for single sign-on: one login for many apps.",
    "SAML is an XML standard for exchanging sign-in assertions.",
    "SCIM: System for Cross-domain Identity Management.",
    "Exponential backoff waits longer after each failed retry.",
    "Parquet is a columnar file format for analytics data.",
    "A SOC 2 report describes a company's security controls.",
    "TLS encrypts data in transit between two systems.",
    "An API key is a secret that identifies a calling program.",
    "OAuth lets an app act for a user without their password.",
    "Schema drift is a change in the structure of source data.",
)

COMPANY = re.compile(
    r"larkspur|\bour\b|\bmy\b|\bwe\b|\bi\b|workspace|\bplan\b"
    r"|starter|team|business|enterprise|lk-\d|release|pipeline"
    r"|refund|price|cost|trial|invoice|region|support",
    re.I,
)


def training_memory() -> list[Passage]:
    """General facts plus the superseded company documents."""
    stale = [d for d in load_corpus() if d.status == "superseded"]
    general = Passage("memory#general", "\n".join(GENERAL_FACTS))
    return [general] + [
        Passage(f"memory#{d.id}", d.text, d.title) for d in stale
    ]


def closed_book(question: str, model: str = "small") -> Answer:
    """Answer from parametric knowledge only, with no search."""
    answer = generate(question, training_memory(), MODELS[model])
    tokens_in = 120 + count_tokens(question)  # no context sent
    if not answer.used:
        return Answer(GUESS, tokens_in=tokens_in, tokens_out=10)
    return Answer(
        answer.text,
        (),
        (),
        False,
        False,
        tokens_in,
        answer.tokens_out,
    )


DEFINITION = re.compile(
    r"^what (is|are|does) (an? )?[\w -]+?( stand for| mean)?\?$",
    re.I,
)


def needs_retrieval(question: str) -> bool:
    """Search unless it is a general definition question."""
    general = DEFINITION.match(question.strip())
    return not general or bool(COMPANY.search(question))


GENERAL_QUESTIONS = (
    ("What does SSO stand for?", "single sign-on"),
    ("What does SCIM stand for?", "Cross-domain Identity"),
    ("What is exponential backoff?", "waits longer"),
    ("What is Parquet?", "columnar"),
    ("What is a SOC 2 report?", "security controls"),
    ("What is TLS?", "in transit"),
    ("What is an API key?", "calling program"),
    ("What is OAuth?", "without their password"),
    ("What is schema drift?", "structure of source data"),
    ("What is SAML?", "sign-in assertions"),
)


@dataclass
class Routed:
    question: str
    searched: bool
    answer: Answer


class Assistant:
    """Answers under a policy: always, never or router."""

    def __init__(
        self, pipeline: Pipeline, policy: str = "router"
    ):
        self.pipeline = pipeline
        self.policy = policy

    def ask(
        self, question: str, user: User | None = None
    ) -> Routed:
        search = {
            "always": True,
            "never": False,
            "router": needs_retrieval(question),
        }[self.policy]
        if not search:
            model = self.pipeline.config.model
            return Routed(
                question, False, closed_book(question, model)
            )
        result = self.pipeline.run(question, user)
        return Routed(question, True, result.answer)


VOCAB = {
    "plan": ("Starter", "Team", "Business", "Enterprise"),
    "role": ("Owners", "Admins", "Editors", "Viewers"),
    "release": ("4.4", "4.5", "4.6"),
}
DESCRIPTOR = re.compile(
    r"\b(?:the|every) (?:cheapest )?(plan|role|release)s?\b[^,?]*"
)
ONE = re.compile(r"the (cheaper|cheapest|first) one")


@dataclass(frozen=True)
class Plan:
    """A first question, and a template for the second."""

    first: str
    then: str  # contains {x}, filled with the bridge entity
    kind: str


def decompose(question: str) -> Plan | None:
    """Split a chained question; None if it is not chained."""
    if ", and " in question:
        head, tail = question.rstrip("?").split(", and ", 1)
        kind = next((k for k in VOCAB if k in head.lower()), "")
        if kind and ONE.search(tail):
            then = ONE.sub("{x}", tail)
            return Plan(
                head + "?", then[0].upper() + then[1:] + "?", kind
            )
    m = DESCRIPTOR.search(question)
    if not m:
        return None
    kind = m.group(1)
    rest = re.split(rf"{kind}s?\b", m.group(0), maxsplit=1)[1]
    rest = rest.strip()
    rest = re.sub(r"^(that|which|who) ", "", rest)
    then = question.replace(m.group(0), "{x}")
    return Plan(f"Which {kind} {rest}?", then, kind)


def name(kind: str, entity: str) -> str:
    """How a bridge entity reads inside the follow-up question."""
    return {
        "plan": f"the {entity} plan",
        "release": f"release {entity}",
    }.get(kind, entity)


def bridges(kind: str, text: str, question: str) -> list[str]:
    """Entities of the wanted kind, in the order they appear."""
    found = [
        (text.find(e), e)
        for e in VOCAB[kind]
        if e in text and e not in question
    ]
    return [e for _, e in sorted(found)]


@dataclass
class Hop:
    query: str
    answer: Answer


@dataclass
class MultiHopResult:
    question: str
    answer: Answer
    hops: list[Hop] = field(default_factory=list)


def combine(answers: list[Answer], all_hops: list[Hop]) -> Answer:
    """Merge follow-up answers; count tokens from every hop."""
    texts = list(dict.fromkeys(a.text for a in answers))
    used = tuple(
        dict.fromkeys(u for a in answers for u in a.used)
    )
    cites = tuple(
        dict.fromkeys(c for a in answers for c in a.citations)
    )
    return Answer(
        " ".join(texts),
        used,
        cites,
        not answers,
        any(a.injected for a in answers),
        sum(h.answer.tokens_in for h in all_hops),
        sum(h.answer.tokens_out for h in all_hops),
    )


class MultiHopAgent:
    """Search, read, then search again using what it found."""

    def __init__(self, pipeline: Pipeline, max_bridges: int = 2):
        self.pipeline = pipeline
        self.max_bridges = max_bridges

    def run(self, question: str, user: User | None = None):
        user = user or user_for("customer")
        plan = decompose(question)
        if plan is None:
            r = self.pipeline.run(question, user)
            return MultiHopResult(
                question, r.answer, [Hop(question, r.answer)]
            )
        first = self.pipeline.run(plan.first, user)
        hops = [Hop(plan.first, first.answer)]
        headers = [
            c.header
            for c in first.context
            if c.id in first.answer.used
        ]
        evidence = " ".join([first.answer.text, *headers])
        found = bridges(plan.kind, evidence, question)
        if first.answer.abstained or not found:
            return MultiHopResult(question, first.answer, hops)
        answers = []
        for entity in found[: self.max_bridges]:
            follow = plan.then.format(x=name(plan.kind, entity))
            r = self.pipeline.run(follow, user)
            hops.append(Hop(follow, r.answer))
            if not r.answer.abstained:
                answers.append(r.answer)
        return MultiHopResult(
            question, combine(answers, hops), hops
        )
