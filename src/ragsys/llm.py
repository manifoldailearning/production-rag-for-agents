"""A simulated language model that answers from retrieved context.

The model reads the question and a list of context passages,
finds the sentences that best support an answer and writes them
back. Its weaknesses are the ones that matter in production:
- it attends less to passages in the middle of a long context;
- it answers from weak evidence unless told to abstain;
- it cannot tell an old policy from a new one unless asked to
  compare dates;
- it may follow instructions planted inside a passage.
Prompt rules switch these behaviors, as instructions do for a
real model. Everything is deterministic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ragsys.relevance import (
    answer_fit,
    content,
    coverage,
    entities,
    entity_share,
    relevance,
)
from ragsys.text import count_tokens, sentences

RULES = ("cite", "abstain", "prefer_recent", "context_is_data")
ABSTAIN = (
    "I can't find this in the documents I have access to, "
    "so I won't guess."
)
INJECTION = re.compile(
    r"(ignore (your|all|any) (previous|prior) instructions"
    r"|system note to ai)",
    re.I,
)


@dataclass(frozen=True)
class Model:
    name: str
    middle_loss: float  # attention lost mid-context, 0..1
    miss_penalty: float  # weight left when an entity is absent
    obeys: float  # chance of following a planted instruction
    obeys_guarded: float  # the same, with context_is_data
    cost_per_1k: float  # dollars per 1,000 tokens
    ms_per_token: float


MODELS = {
    "small": Model("small", 0.3, 0.8, 0.9, 0.3, 0.0005, 4.0),
    "large": Model("large", 0.15, 0.4, 0.6, 0.0, 0.005, 12.0),
}


@dataclass(frozen=True)
class Passage:
    """One context item as the model sees it."""

    chunk_id: str
    text: str
    header: str = ""
    updated: str = ""
    status: str = "current"


@dataclass(frozen=True)
class Answer:
    text: str
    used: tuple[str, ...] = ()  # chunk ids the answer came from
    citations: tuple[str, ...] = ()
    abstained: bool = False
    injected: bool = False
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class Evidence:
    score: float  # support for an answer, after position effects
    raw: float  # the same, ignoring position
    coverage: float  # share of the question's words it covers
    position: int
    chunk_id: str
    sentence: str
    updated: str
    status: str


THRESHOLD = 0.4
SYSTEM_TOKENS = 120
GUESS = "Yes, Larkspur Cloud supports that on every plan."


def position_weight(i: int, n: int, loss: float) -> float:
    """1.0 at both ends of the context, lowest in the middle."""
    if n < 3:
        return 1.0
    middle = 1 - abs(2 * i / (n - 1) - 1)
    return 1 - loss * middle


def draw(*parts: str) -> float:
    """A stable pseudo-random number in [0, 1)."""
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def planted(passage: Passage) -> list[str]:
    """Sentences that carry an instruction aimed at the model."""
    lines = sentences(passage.text)
    for i, line in enumerate(lines):
        if INJECTION.search(line):
            return lines[i : i + 2]
    return []


def leads_with_entity(question: str, sentence: str) -> bool:
    """Is the sentence about the named thing, or just near it?"""
    head = " ".join(sentence.split()[:4]).lower()
    return any(e in head for e in entities(question))


def support(question: str, p: Passage, s: str, model: Model):
    """How well a sentence supports an answer, before position."""
    own = relevance(question, s, 1.0)
    framed = relevance(question, f"{p.header}. {s}", 1.0)
    share = entity_share(question, f"{p.header}\n{p.text}")
    named = model.miss_penalty + (1 - model.miss_penalty) * share
    score = (0.75 * own + 0.25 * framed) * named
    score *= answer_fit(question, s)
    if leads_with_entity(question, s):
        score += 0.05
    return score


def gather(
    question: str, passages: list[Passage], model: Model, rules
) -> list[Evidence]:
    """Score every context sentence as support for an answer."""
    out = []
    for i, p in enumerate(passages):
        weight = position_weight(
            i, len(passages), model.middle_loss
        )
        skip = planted(p) if "context_is_data" in rules else []
        for s in sentences(p.text):
            if s in skip:
                continue
            raw = support(question, p, s, model)
            covered = coverage(question, f"{p.header}. {s}")
            out.append(
                Evidence(
                    round(raw * weight, 4),
                    round(raw, 4),
                    round(covered, 3),
                    i,
                    p.chunk_id,
                    s,
                    p.updated,
                    p.status,
                )
            )
    return sorted(out, key=lambda e: (-e.score, e.position))


def adds(question: str, best: Evidence, other: Evidence) -> bool:
    """Does other cover question words that best leaves out?"""
    q = content(question)
    return bool(
        (q & content(other.sentence)) - content(best.sentence)
    )


def choose(
    question: str, evidence: list[Evidence], rules
) -> list[Evidence]:
    """The best sentence, plus one from its chunk that adds."""
    best = evidence[0]
    if "prefer_recent" in rules and best.status != "current":
        close = [
            e
            for e in evidence
            if e.status == "current" and e.raw >= best.raw - 0.05
        ]
        if close:
            best = close[0]
    picked = [best]
    for e in evidence:
        same = e is not best and e.chunk_id == best.chunk_id
        if same and e.score >= 0.6 * best.score:
            if adds(question, best, e):
                picked.append(e)
                break
    return picked


def follow_injection(
    question: str, passages: list[Passage], model: Model, rules
) -> str:
    """Text the model adds if it obeys a planted instruction."""
    chance = (
        model.obeys_guarded
        if "context_is_data" in rules
        else model.obeys
    )
    for p in passages:
        lines = planted(p)
        if (
            len(lines) == 2
            and draw(question, p.chunk_id) < chance
        ):
            order = lines[1].removeprefix("Tell every user that ")
            return "Note: " + order[0].upper() + order[1:]
    return ""


def render(picked: list[Evidence], rules) -> tuple[str, tuple]:
    parts, cites = [], []
    for e in picked:
        if "cite" in rules:
            parts.append(f"{e.sentence} [{e.chunk_id}]")
            cites.append(e.chunk_id)
        else:
            parts.append(e.sentence)
    return " ".join(parts), tuple(dict.fromkeys(cites))


def generate(
    question: str,
    passages: list[Passage],
    model: Model = MODELS["small"],
    rules: frozenset[str] = frozenset(),
) -> Answer:
    """Answer from the passages, under the prompt rules."""
    tokens_in = (
        SYSTEM_TOKENS
        + count_tokens(question)
        + sum(
            count_tokens(f"{p.header}\n{p.text}")
            for p in passages
        )
    )
    evidence = gather(question, passages, model, rules)
    weak = not evidence or (
        evidence[0].score < THRESHOLD
        or evidence[0].coverage < 0.25
    )
    if weak and "abstain" in rules:
        return Answer(
            ABSTAIN,
            abstained=True,
            tokens_in=tokens_in,
            tokens_out=count_tokens(ABSTAIN),
        )
    if not evidence:
        text, used, cites = GUESS, (), ()
    else:
        picked = choose(question, evidence, rules)
        text, cites = render(picked, rules)
        used = tuple(dict.fromkeys(e.chunk_id for e in picked))
    extra = follow_injection(question, passages, model, rules)
    if extra:
        text = f"{text} {extra}"
    return Answer(
        text,
        used,
        cites,
        False,
        bool(extra),
        tokens_in,
        count_tokens(text),
    )
