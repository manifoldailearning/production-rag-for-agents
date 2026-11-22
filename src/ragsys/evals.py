"""Grade a pipeline against the question set.

Every question has gold documents, strings a correct answer
must contain and strings it must not. Unanswerable and
restricted questions are correct only when the system abstains.
Retrieval and generation are graded separately, so a failure
can be traced to the stage that caused it.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources

from ragsys.access import can_read, user_for
from ragsys.pipeline import Pipeline, Result

ABSTAIN_TYPES = {"unanswerable", "restricted"}


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    gold: tuple[str, ...]
    expected: tuple[tuple[str, ...], ...]
    wrong: tuple[str, ...]
    user: str = "customer"
    tenant: str = ""

    @property
    def should_abstain(self) -> bool:
        return self.type in ABSTAIN_TYPES


def load_questions(
    name: str = "questions.json",
) -> list[Question]:
    path = resources.files("ragsys") / "data" / name
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            r["id"],
            r["type"],
            r["question"],
            tuple(r["gold"]),
            tuple(tuple(g) for g in r["expected"]),
            tuple(r["wrong"]),
            r["user"],
            r["tenant"],
        )
        for r in rows
    ]


def split(questions: list[Question], part: str) -> list[Question]:
    """A fixed dev/test split; tuning never sees test answers."""

    def is_test(q: Question) -> bool:
        return (
            int(hashlib.sha256(q.id.encode()).hexdigest(), 16)
            % 10
            < 3
        )

    return [
        q for q in questions if is_test(q) == (part == "test")
    ]


def is_correct(q: Question, text: str, abstained: bool) -> bool:
    """Contains every expected fact and none of the wrong ones."""
    if q.should_abstain:
        return abstained
    if abstained:
        return False
    text = text.lower()
    has_all = all(
        any(a.lower() in text for a in g) for g in q.expected
    )
    return has_all and not any(w.lower() in text for w in q.wrong)


@dataclass(frozen=True)
class Grade:
    id: str
    type: str
    correct: bool
    found: bool  # a gold document in the top k results
    in_context: bool  # a gold document reached the model
    abstained: bool
    leaked: bool  # the answer used a document the user can't read
    exposed: bool  # such a document reached the context
    injected: bool
    tokens_in: int
    tokens_out: int


def grade(q: Question, r: Result, pipe: Pipeline) -> Grade:
    gold = set(q.gold)
    top_k = pipe.config.top_k
    hit_docs = {h.chunk.doc_id for h in r.hits[:top_k]}
    ctx_docs = {c.doc_id for c in r.context}
    used_docs = {cid.split("#")[0] for cid in r.answer.used}

    def unreadable(ids: set[str]) -> bool:
        return any(
            not can_read(r.user, pipe.by_id[i]) for i in ids
        )

    return Grade(
        q.id,
        q.type,
        is_correct(q, r.answer.text, r.answer.abstained),
        bool(gold & hit_docs),
        bool(gold & ctx_docs),
        r.answer.abstained,
        unreadable(used_docs),
        unreadable(ctx_docs),
        r.answer.injected,
        r.answer.tokens_in,
        r.answer.tokens_out,
    )


def evaluate(pipe: Pipeline, questions=None) -> list[Grade]:
    """Run every question as its own user and grade the result."""
    grades = []
    for q in questions or load_questions():
        user = user_for(q.user, q.tenant)
        grades.append(grade(q, pipe.run(q.question, user), pipe))
    return grades


def rate(flags) -> float:
    flags = list(flags)
    return round(sum(flags) / len(flags), 3) if flags else 0.0


def summarize(grades: list[Grade]) -> dict[str, float]:
    answerable = [
        g for g in grades if g.type not in ABSTAIN_TYPES
    ]
    return {
        "accuracy": rate(g.correct for g in grades),
        "recall": rate(g.found for g in answerable),
        "context_recall": rate(g.in_context for g in answerable),
        "leaks": sum(g.leaked for g in grades),
        "exposed": sum(g.exposed for g in grades),
        "injected": sum(g.injected for g in grades),
        "tokens_in": round(
            sum(g.tokens_in for g in grades) / len(grades)
        ),
    }


def by_type(grades: list[Grade]) -> dict[str, tuple[int, int]]:
    """Correct and total per question type."""
    table: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for g in grades:
        table[g.type][0] += g.correct
        table[g.type][1] += 1
    return {t: (c, n) for t, (c, n) in sorted(table.items())}
