"""Chapter 16: GraphRAG in miniature, with nothing but Python.

Seven one-line documents describe who leads which team, which
team owns which service and which service depends on which
system. The question "Who must be told if Postgres goes down?"
needs three hops through them. Keyword search finds the passages
that mention Postgres and none of the people; a graph built from
the same documents follows the chain and cites every link.
"""

from __future__ import annotations

import re
from collections import defaultdict

DOCS = {
    "d1": "Priya leads the Payments team.",
    "d2": "The Payments team owns the Billing API.",
    "d3": "The Billing API depends on Postgres.",
    "d4": "Arjun leads the Search team.",
    "d5": "The Search team owns the Catalog service.",
    "d6": "The Catalog service depends on Postgres.",
    "d7": "The Catalog service depends on Redis.",
}
FACT = re.compile(
    r"^(?:The )?(.+?) (leads|owns|depends on) (?:the )?(.+?)\.$"
)


def extract(docs: dict[str, str]) -> list[tuple]:
    """Stand-in for a model: (subject, relation, object, doc)."""
    triples = []
    for doc_id, text in docs.items():
        m = FACT.match(text)
        if m:
            triples.append((*m.groups(), doc_id))
    return triples


def build(triples: list[tuple]) -> dict[str, list[tuple]]:
    """Index edges by their object, to walk toward dependents."""
    into = defaultdict(list)
    for subject, relation, obj, doc_id in triples:
        into[obj].append((subject, relation, doc_id))
    return into


def paths(graph, node: str, path=()) -> list[tuple]:
    """Every chain of edges that ends at node."""
    found = []
    for subject, relation, doc_id in graph.get(node, []):
        step = ((subject, relation, node, doc_id),) + path
        found.append(step)
        found += paths(graph, subject, step)
    return found


def who_to_tell(graph, system: str):
    """People who lead a team whose service needs the system."""
    chains = [
        p for p in paths(graph, system) if p[0][1] == "leads"
    ]
    people = [p[0][0] for p in chains]
    docs = sorted({step[3] for p in chains for step in p})
    return people, chains, docs


def keyword_search(question: str, k: int = 2) -> list[str]:
    """Rank documents by words shared with the question."""
    words = set(re.findall(r"\w+", question.lower()))

    def score(doc_id: str) -> int:
        text = DOCS[doc_id].lower()
        return len(words & set(re.findall(r"\w+", text)))

    return sorted(DOCS, key=lambda d: (-score(d), d))[:k]


def main() -> None:
    triples = extract(DOCS)
    print(f"extracted {len(triples)} triples")
    for s, rel, o, doc_id in triples[:3]:
        print(f"  ({s}) -[{rel}]-> ({o})  [{doc_id}]")
    print("  ...")

    question = "Who must be told if Postgres goes down?"
    print(f"\n{question}")
    top = keyword_search(question)
    named = [
        n
        for n in ("Priya", "Arjun")
        if any(n in DOCS[d] for d in top)
    ]
    print(f"keyword search, top 2: {', '.join(top)}")
    print(f"  people in those passages: {named or 'none'}")

    graph = build(triples)
    people, chains, docs = who_to_tell(graph, "Postgres")
    print("graph traversal:")
    for chain in chains:
        names = [chain[0][0]] + [step[2] for step in chain]
        print("  " + " -> ".join(names))
    print(f"answer: {' and '.join(people)}  [{', '.join(docs)}]")

    users = [
        s
        for s, rel, _ in graph["Postgres"]
        if rel == "depends on"
    ]
    print(f"\nWhat depends on Postgres? {', '.join(users)}")


if __name__ == "__main__":
    main()
