"""Chapter 14: planted instructions and personal data."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.configs import CONFIGS, GROUNDED, GUARDED
from ragsys.corpus import load_corpus
from ragsys.evals import evaluate, load_questions, summarize
from ragsys.pipeline import Pipeline
from ragsys.safety import clean, guard_output

BASE = CONFIGS["reranked"].but(metadata=True, acl=True)


def main() -> None:
    docs = load_corpus()
    cleaned, findings = clean(docs)
    for f in findings:
        print(f"{f.kind:11} {f.doc_id:22} {f.text[:28]}")
    questions = load_questions()
    print(f"\n{'defence':24} {'obeyed':>6} {'accuracy':>9}")
    rows = (
        ("none", BASE.but(rules=GROUNDED), docs),
        ("context_is_data rule", BASE.but(rules=GUARDED), docs),
        (
            "rule + verification",
            BASE.but(rules=GUARDED, verify=True),
            docs,
        ),
        (
            "quarantine at ingest",
            BASE.but(rules=GROUNDED),
            cleaned,
        ),
    )
    for name, cfg, corpus in rows:
        s = summarize(evaluate(Pipeline(cfg, corpus), questions))
        print(f"{name:24} {s['injected']:6} {s['accuracy']:9.1%}")
    q = "Who was the customer contact on ticket 88412?"
    staff = user_for("support")
    for name, corpus in (("raw", docs), ("redacted", cleaned)):
        text = (
            Pipeline(BASE.but(rules=GUARDED), corpus)
            .run(q, staff)
            .answer.text
        )
        print(f"\n{name}: {textwrap.shorten(text, 54)}")
    blocked = guard_output(
        "Email billing-help@larkspur-support.example"
    )
    print(f"output guard blocked: {blocked[1]}")


if __name__ == "__main__":
    main()
