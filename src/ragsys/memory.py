"""Memory for a conversational agent.

Two different problems share the word memory. Short-term memory
turns a follow-up such as "what about Business?" into a question
the index can answer on its own. Long-term memory keeps facts a
user stated in earlier sessions, such as their plan, and must
handle updates and deletion, or it answers from stale facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ragsys.agent import VOCAB

PLANS = VOCAB["plan"]
FOLLOW_UP = re.compile(
    r"^(?:what|how) about (?:the )?(\w+)(?: plan)?\??$|"
    r"^and (?:on|for) (?:the )?(\w+)(?: plan)?\??$",
    re.I,
)


def follow_up_entity(question: str) -> str | None:
    m = FOLLOW_UP.match(question.strip())
    if not m:
        return None
    word = (m.group(1) or m.group(2)).capitalize()
    return word if word in PLANS else None


def contextualize(question: str, history: list[str]) -> str:
    """Rewrite a follow-up into a standalone question."""
    entity = follow_up_entity(question)
    if not entity or not history:
        return question
    previous = history[-1]
    for plan in PLANS:
        if plan in previous:
            return previous.replace(plan, entity)
    return f"{previous} ({entity} plan)"


def concatenate(question: str, history: list[str]) -> str:
    """The naive alternative: glue the last question on."""
    return f"{history[-1]} {question}" if history else question


DIALOGUES = (
    (
        (
            "How long are pipeline logs kept on the Team plan?",
            "30 days",
        ),
        ("What about Business?", "90 days"),
        ("And for Enterprise?", "365 days"),
    ),
    (
        ("What does the Starter plan cost per month?", "$49"),
        ("What about Team?", "$299"),
        ("And for Business?", "$1,200"),
    ),
    (
        (
            "What is the minimum schedule interval on Starter?",
            "60 minutes",
        ),
        ("What about Team?", "15 minutes"),
    ),
    (
        (
            "Is single sign-on available on the Starter plan?",
            "not available",
        ),
        ("What about Team?", "add"),
    ),
    (
        (
            "What uptime is committed on the Team plan?",
            "99.5",
        ),
        ("What about Enterprise?", "99.95"),
    ),
)

STATED_PLAN = re.compile(
    r"\b(?:we're|we are|i'm|i am|moved|upgraded|switched)\b"
    r"[^.]*?\b(Starter|Team|Business|Enterprise)\b",
    re.I,
)
FORGET = re.compile(r"\bforget\b[^.]*\bplan\b", re.I)


@dataclass(frozen=True)
class Fact:
    slot: str
    value: str
    turn: int


@dataclass
class MemoryStore:
    """Long-term facts per user, under a write policy.

    append: keep every statement; recall returns the first match.
    slots: one value per slot; a newer statement replaces it, and
    an explicit request to forget removes it.
    """

    policy: str = "slots"
    facts: dict[str, list[Fact]] = field(default_factory=dict)

    def observe(self, user: str, text: str, turn: int) -> None:
        mine = self.facts.setdefault(user, [])
        if FORGET.search(text) and self.policy == "slots":
            mine[:] = [f for f in mine if f.slot != "plan"]
            return
        m = STATED_PLAN.search(text)
        if not m:
            return
        fact = Fact("plan", m.group(1).capitalize(), turn)
        if self.policy == "slots":
            mine[:] = [f for f in mine if f.slot != fact.slot]
        mine.append(fact)

    def recall(self, user: str, slot: str) -> str | None:
        found = [
            f for f in self.facts.get(user, []) if f.slot == slot
        ]
        return found[0].value if found else None


PERSONAL = re.compile(r"\b(my|our)\b", re.I)


def personalize(question: str, plan: str | None) -> str:
    """Make a question about "my plan" explicit, if we know it."""
    if not plan or not PERSONAL.search(question):
        return question
    if any(p in question for p in PLANS):
        return question
    q = re.sub(
        r"\b(?:on )?(my|our) plan\b",
        f"on the {plan} plan",
        question,
    )
    if q == question:
        q = question.rstrip("?") + f" on the {plan} plan?"
    return q


SESSIONS = (
    (
        ("We're on the Team plan.",),
        "How long are my logs kept?",
        "30 days",
    ),
    (
        (
            "We're on the Team plan.",
            "We upgraded to the Business plan last week.",
        ),
        "How long are my logs kept?",
        "90 days",
    ),
    (
        ("I'm on the Starter plan.",),
        "Can I use single sign-on on my plan?",
        "not available",
    ),
    (
        (
            "We're on the Team plan.",
            "Actually we switched to Enterprise.",
        ),
        "What is the monthly price of our plan?",
        "Custom",
    ),
    (
        ("We moved to Enterprise.", "Please forget my plan."),
        "What is my support response target?",
        None,
    ),
)
