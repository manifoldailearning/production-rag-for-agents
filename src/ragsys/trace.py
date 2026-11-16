"""Find the stage where a wrong answer went wrong.

When an answer is wrong, the useful question is where. The gold
document may never have been retrieved, retrieved but ranked too
low, ranked high but squeezed out of the context, or delivered
to the model and still misread. Each cause has a different fix,
so each failure is labeled with one stage.
"""

from __future__ import annotations

from collections import Counter

from ragsys.access import user_for
from ragsys.evals import Question, is_correct
from ragsys.pipeline import Pipeline, Result

STAGES = ("ok", "retrieval", "ranking", "context", "generation")


def diagnose(q: Question, r: Result, pipe: Pipeline) -> str:
    """Label a result with the first stage that lost it."""
    if is_correct(q, r.answer.text, r.answer.abstained):
        return "ok"
    if q.should_abstain or not q.gold:
        return "generation"
    gold = set(q.gold)
    ranked = [h.chunk.doc_id for h in r.hits]
    in_context = {c.doc_id for c in r.context}
    if not gold & set(ranked):
        return "retrieval"
    if not gold & set(ranked[: pipe.config.top_k]):
        return "ranking"
    if not gold & in_context:
        return "context"
    return "generation"


def failure_profile(pipe: Pipeline, questions) -> Counter[str]:
    counts: Counter[str] = Counter()
    for q in questions:
        r = pipe.run(q.question, user_for(q.user, q.tenant))
        counts[diagnose(q, r, pipe)] += 1
    return counts


def to_record(r: Result, pipe: Pipeline) -> dict:
    """One structured log record per request, for replay."""
    return {
        "config": pipe.config.name,
        "user": r.user.name,
        "tenant": r.user.tenant,
        "question": r.question,
        "query": r.query,
        "hits": [
            [h.chunk.id, round(h.score, 3)]
            for h in r.hits[: pipe.config.top_k]
        ],
        "context": [c.id for c in r.context],
        "cited": list(r.answer.citations),
        "abstained": r.answer.abstained,
        "tokens": [r.answer.tokens_in, r.answer.tokens_out],
    }
