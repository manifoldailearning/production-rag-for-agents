"""Chapter 16: when a graph pays off, and what it costs."""

from __future__ import annotations

from ragsys.access import can_read, user_for
from ragsys.agent import MultiHopAgent
from ragsys.configs import PRODUCTION
from ragsys.evals import is_correct, load_questions
from ragsys.graph import (
    GOLD,
    assistant,
    build,
    check,
    extraction_cost,
)
from ragsys.pipeline import Pipeline


def score(questions, ask) -> tuple[dict, int]:
    """Correct answers per question type, and mean tokens in."""
    ok: dict[str, int] = {}
    tokens = 0
    for q in questions:
        a = ask(q.question, user_for(q.user, q.tenant))
        ok[q.type] = ok.get(q.type, 0) + is_correct(
            q, a.text, a.abstained
        )
        tokens += a.tokens_in
    return ok, round(tokens / len(questions))


def main() -> None:
    router = assistant("router")
    graph_only = assistant("graph")
    pipe = router.pipeline
    g = router.graph
    counts = ", ".join(f"{r} {n}" for r, n in g.counts().items())
    print(
        f"graph: {len(g.entities())} entities, {len(g.edges)} "
        f"relations from {len(g.chunks)} chunks"
    )
    print(f"  {counts}")
    tin, tout, dollars = extraction_cost(g)
    print(
        f"extraction: {tin:,} tokens in, {tout:,} out, ${dollars}"
    )

    customer = user_for("customer")
    full = Pipeline(PRODUCTION.but(drop_superseded=False))
    head = f"plan facts vs {len(GOLD)} labeled"
    print(f"\n{head:30}precision  recall")
    for schema, label in (
        (1, "two relations"),
        (2, "with add-on"),
    ):
        view = build(full, schema).visible(customer, full.by_id)
        p, r, wrong = check(view)
        print(f"  {label:28}{p:9.1%}{r:8.1%}")
    print("  still wrong:")
    for w in wrong:
        print("    " + " ".join(w))

    hop = MultiHopAgent(pipe)
    systems = (
        ("search", lambda q, u: pipe.run(q, u).answer),
        ("multi-hop agent", lambda q, u: hop.run(q, u).answer),
        ("graph router", lambda q, u: router.ask(q, u).answer),
    )
    graph_qs = load_questions("graph_questions.json")
    print(
        "\n10 graph questions   aggregate  relationship  tokens"
    )
    for name, ask in systems:
        ok, tokens = score(graph_qs, ask)
        agg, rel = (
            ok.get("aggregate", 0),
            ok.get("relationship", 0),
        )
        print(f"  {name:19}{agg:7}/6{rel:12}/4{tokens:8}")

    every = load_questions()
    print(f"\n{'90 original questions':29}correct  tokens")
    for name, ask in (
        systems[0],
        ("graph only", lambda q, u: graph_only.ask(q, u).answer),
        systems[2],
    ):
        ok, tokens = score(every, ask)
        right = f"{sum(ok.values())}/90"
        print(f"  {name:27}{right:>7}{tokens:8}")

    team = "Which features are not available on the Team plan?"
    print()
    for label, a in (
        ("stale", assistant("router", fresh=False)),
        ("current", router),
    ):
        r = a.ask(team)
        print(f"Team lacks, {label + ':':9}{', '.join(r.items)}")
        print(f"{'':21}from {', '.join(r.answer.used)}")
    stripe = (
        "Which error codes were seen with the Stripe connector?"
    )
    for label, a in (
        ("unfiltered", assistant("router", permissions=False)),
        ("filtered", router),
    ):
        r = a.ask(stripe, customer)
        leaked = any(
            not can_read(customer, pipe.by_id[c.split("#")[0]])
            for c in r.answer.used
        )
        found = ", ".join(r.items) or "abstained"
        flag = "LEAKED" if leaked else "no leak"
        print(f"Stripe errors, {label + ':':12}{found:12}{flag}")


if __name__ == "__main__":
    main()
