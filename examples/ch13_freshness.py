"""Chapter 13: stale documents, updates and deletions."""

from __future__ import annotations

import textwrap
from dataclasses import replace

from ragsys.configs import CONFIGS, PRODUCTION
from ragsys.corpus import load_corpus
from ragsys.evals import by_type, evaluate, load_questions
from ragsys.freshness import DocStore, IncrementalIndex
from ragsys.pipeline import Pipeline

TEAM = "What does the Team plan cost per month?"
RECENT = frozenset({"cite", "prefer_recent"})


def temporal(cfg) -> str:
    ok, n = by_type(evaluate(Pipeline(cfg), load_questions()))[
        "temporal"
    ]
    return f"{ok}/{n}"


def show(label: str, r) -> None:
    print(
        f"{label:11} added {r.added}, changed {r.changed}, "
        f"removed {r.removed}, embedded {r.embedded_chunks}"
    )


def main() -> None:
    base = CONFIGS["reranked"].but(rules=frozenset({"cite"}))
    print("temporal questions answered correctly:")
    print(f"  index everything       {temporal(base)}")
    print(
        f"  + dates, prefer_recent "
        f"{temporal(base.but(metadata=True, rules=RECENT))}"
    )
    print(
        f"  drop superseded docs   "
        f"{temporal(base.but(drop_superseded=True))}"
    )
    docs = load_corpus()
    store = DocStore({d.id: d for d in docs})
    index = IncrementalIndex()
    show("first sync", index.sync(store))
    v3 = store.docs["pricing-v3"]
    v4 = replace(
        v3,
        id="pricing-v4",
        updated="2026-11-16",
        text=v3.text.replace("$299 per month", "$329 per month"),
    )
    stale = Pipeline(
        PRODUCTION, list(store.docs.values()), index.index
    )
    store.supersede("pricing-v3", v4)
    show("after edit", index.sync(store))
    fresh = Pipeline(
        PRODUCTION, list(store.docs.values()), index.index
    )
    for label, pipe in (
        ("stale index", stale),
        ("synced", fresh),
    ):
        text = textwrap.shorten(pipe.run(TEAM).answer.text, 40)
        print(f"  {label:12} {text}")
    store.delete("community-postgres-tip")
    for deletes in (False, True):
        ix = IncrementalIndex(handle_deletes=deletes)
        ix.chunks = dict(index.chunks)
        ix.seen = dict(index.seen)
        ix.vectors = dict(index.vectors)
        ix.sync(store)
        left = any("community" in c.id for c in ix.all_chunks())
        print(
            f"handle_deletes={deletes!s:5}: deleted post still "
            f"indexed: {left}"
        )


if __name__ == "__main__":
    main()
