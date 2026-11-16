"""Chapter 12: permissions, tenants and the cache that leaks."""

from __future__ import annotations

import textwrap

from ragsys.access import user_for
from ragsys.cache import AnswerCache
from ragsys.configs import PRODUCTION
from ragsys.evals import (
    by_type,
    evaluate,
    load_questions,
    summarize,
)
from ragsys.pipeline import Pipeline

MODES = (
    ("no filter", PRODUCTION.but(acl=False)),
    (
        "post-filter",
        PRODUCTION.but(acl_mode="post", candidates=4),
    ),
    ("pre-filter", PRODUCTION),
)


def main() -> None:
    questions = load_questions()
    print(
        f"{'':12} {'leaks':>5} {'exposed':>8} {'tenant':>7} "
        f"{'restricted':>11}"
    )
    for name, cfg in MODES:
        grades = evaluate(Pipeline(cfg), questions)
        s, kinds = summarize(grades), by_type(grades)
        t, r = kinds["tenant"], kinds["restricted"]
        print(
            f"{name:12} {s['leaks']:5} {s['exposed']:8} "
            f"{t[0]:5}/{t[1]} {r[0]:9}/{r[1]}"
        )
    pipe = Pipeline(PRODUCTION)
    q = "How long are pipeline logs kept for our workspaces?"
    orchid = user_for("customer", "orchid")
    bramble = user_for("customer", "bramble")
    for scoped in (False, True):
        cache = AnswerCache(scoped=scoped)
        first = pipe.run(q, orchid).answer
        cache.put(q, orchid, "v1", first)
        hit = cache.get(q, bramble, "v1")
        shown = (
            hit.answer.text
            if hit
            else pipe.run(q, bramble).answer.text
        )
        label = "scoped" if scoped else "unscoped"
        print(f"\n{label} cache, Bramble sees:")
        print(f"  {textwrap.shorten(shown, 60)}")


if __name__ == "__main__":
    main()
