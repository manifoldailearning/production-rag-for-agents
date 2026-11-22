"""Entities and relations extracted from the corpus, as a graph.

Some questions are about how things relate across many documents:
everything the Starter plan lacks, every change in the last three
releases, the plans that can use a connector a release changed.
Chunk search returns a few passages and the model answers from one
or two sentences, so lists spread over many documents come back
incomplete. A graph built at indexing time answers them with a
query instead.

Extraction stands in for a language model reading every chunk:
it finds entities from a fixed schema and the relations between
them, sentence by sentence, and keeps the chunk each relation came
from. That provenance is what lets permissions, freshness and
citations survive the move from text to graph.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ragsys.access import User, can_read, user_for
from ragsys.chunking import Chunk
from ragsys.configs import GUARDED, PRODUCTION
from ragsys.corpus import PUBLIC, Document
from ragsys.llm import (
    ABSTAIN,
    MODELS,
    SYSTEM_TOKENS,
    Answer,
    Passage,
    generate,
)
from ragsys.pipeline import Pipeline
from ragsys.text import count_tokens, sentences

PLANS = ("Starter", "Team", "Business", "Enterprise")
FEATURES = {
    "SSO": r"single sign-on|\bSSO\b",
    "SCIM provisioning": r"\bSCIM\b",
    "audit logs": r"\baudit logs?\b",
    "larger workers": r"\blarger workers?\b",
    "uptime commitment": r"\buptime\b",
    "support tickets": r"\btickets in the support portal",
    "technical account manager": r"technical account manager",
    "community forum": r"\bcommunity forum\b",
    "invoice billing": r"\bby invoice\b",
    "Slack alerts": r"\bSlack\b",
    "schema evolution": r"\bschema evolution\b",
    "bulk API 2.0": r"\bbulk API 2\.0\b",
}
CONNECTORS = (
    "PostgreSQL",
    "MySQL",
    "Salesforce",
    "Stripe",
    "Amazon S3",
    "Google Sheets",
    "Snowflake",
    "BigQuery",
    "Amazon Redshift",
    "Databricks",
)
REGIONS = ("us-east", "eu-west", "ap-south")
RELEASE = re.compile(r"\b(?:release|Release Notes) (4\.\d)\b")
ERROR = re.compile(r"\bLK-\d{4}\b")


@dataclass(frozen=True)
class Edge:
    """One extracted fact, with the chunk it came from."""

    source: str
    relation: str  # includes, excludes, add-on, changed, related
    target: str
    chunk_id: str
    doc_id: str
    sentence: str


def kind(entity: str) -> str:
    if entity in PLANS:
        return "plan"
    if entity in FEATURES:
        return "feature"
    if entity in CONNECTORS:
        return "connector"
    if entity in REGIONS:
        return "region"
    if entity.startswith("LK-"):
        return "error"
    return "release"


def things(text: str) -> list[str]:
    """Features, connectors and regions the text names."""
    found = [
        f
        for f, pat in FEATURES.items()
        if re.search(pat, text, re.I)
    ]
    found += [c for c in CONNECTORS if c in text]
    return found + [r for r in REGIONS if r in text]


def plans_in(text: str) -> list[str]:
    """Plans named, with "X plan or higher" meaning X and up."""
    m = re.search(r"\b(\w+) plan or higher", text)
    if m and m.group(1) in PLANS:
        return list(PLANS[PLANS.index(m.group(1)) :])
    return [p for p in PLANS if re.search(rf"\b{p}\b", text)]


NEGATIVE = re.compile(
    r"\bnot (available|included)\b|\bhas no\b|\bonly\b", re.I
)
ADDON = re.compile(r"\badd\b|\badd-on\b", re.I)


def plan_edges(s: str, schema: int) -> list[tuple]:
    """(plan, relation, thing) facts stated by one sentence.

    Schema 1 knows two relations, includes and excludes. Schema 2
    adds add-on, for features a plan can buy separately.
    """
    plans, found = plans_in(s), things(s)
    if not plans or not found:
        return []
    out = []
    clauses = re.split(r", and |; ", s)
    for p in plans:
        rel = "includes"
        if NEGATIVE.search(s) and not re.search(r"\bonly\b", s):
            rel = "excludes"
        own = next((c for c in clauses if p in c), s)
        if schema >= 2 and ADDON.search(own):
            rel = "add-on"
        out += [(p, rel, t) for t in found]
    if re.search(r"plan or higher|\bonly\b", s):
        out += [
            (p, "excludes", t)
            for p in PLANS
            if p not in plans
            for t in found
        ]
    return out


def subject(doc: Document) -> str:
    """The release or error a whole document is about, if any."""
    m = RELEASE.search(doc.title) or ERROR.search(doc.title)
    if not m:
        return ""
    return m.group(1) if m.lastindex else m.group(0)


def extract(
    chunks: list[Chunk],
    docs: dict[str, Document],
    schema: int = 2,
) -> list[Edge]:
    """Read every chunk once and keep each fact's provenance."""
    edges: list[Edge] = []
    for c in chunks:
        topic = subject(docs[c.doc_id])
        for s in sentences(c.text):

            def add(a, rel, b, s=s, c=c):
                edges.append(Edge(a, rel, b, c.id, c.doc_id, s))

            for a, rel, b in plan_edges(s, schema):
                add(a, rel, b)
            named = things(s) + plans_in(s)
            release = RELEASE.search(s)
            when = release.group(1) if release else ""
            if not when and topic.startswith("4."):
                when = topic
            for x in named if when else []:
                add(when, "changed", x)
            errors = ERROR.findall(s) or (
                [topic] if topic.startswith("LK-") else []
            )
            for e in errors:
                for x in things(s) + plans_in(s):
                    add(e, "related", x)
    return list(dict.fromkeys(edges))


@dataclass
class Graph:
    edges: list[Edge]
    chunks: dict[str, Chunk] = field(default_factory=dict)

    def counts(self) -> Counter:
        return Counter(e.relation for e in self.edges)

    def entities(self) -> set[str]:
        return {e.source for e in self.edges} | {
            e.target for e in self.edges
        }

    def visible(
        self, user: User, docs: dict[str, Document], fresh=True
    ) -> Graph:
        """Only edges from documents this user may read now."""
        keep = [
            e
            for e in self.edges
            if can_read(user, docs[e.doc_id])
            and (not fresh or docs[e.doc_id].status == "current")
        ]
        return Graph(keep, self.chunks)

    def find(self, source=None, relation=None, target=None):
        return [
            e
            for e in self.edges
            if source in (None, e.source)
            and relation in (None, e.relation)
            and target in (None, e.target)
        ]


def build(pipeline: Pipeline, schema: int = 2) -> Graph:
    """Extract a graph from the pipeline's chunks (all docs)."""
    edges = extract(pipeline.chunks, pipeline.by_id, schema)
    return Graph(edges, {c.id: c for c in pipeline.chunks})


EXTRACT_PROMPT = 400  # schema, instructions and examples
TOKENS_PER_EDGE = 18


def extraction_cost(graph: Graph, model: str = "small") -> tuple:
    """Tokens in, tokens out and dollars to build it once."""
    tin = sum(
        EXTRACT_PROMPT + count_tokens(c.indexed_text)
        for c in graph.chunks.values()
    )
    tout = TOKENS_PER_EDGE * len(graph.edges)
    dollars = (tin + tout) / 1000 * MODELS[model].cost_per_1k
    return tin, tout, round(dollars, 4)


def link(question: str) -> list[str]:
    """Entities a question names, in schema terms."""
    named = things(question) + plans_in(question)
    named += [
        f"LK-{n}" for n in re.findall(r"LK-(\d{4})", question)
    ]
    m = re.search(r"\breleases? (4\.\d)", question)
    return named + ([m.group(1)] if m else [])


@dataclass
class GraphAnswer:
    shape: str  # which query the question became, or "" if none
    items: list[str]
    answer: Answer


def listing(items: list[str]) -> str:
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def respond(
    question: str, shape: str, lead: str, items, edges, detail=""
) -> GraphAnswer:
    """Write the answer from query results, citing every edge."""
    if not items:
        empty = Answer(ABSTAIN, abstained=True)
        return GraphAnswer(shape, [], empty)
    cites = tuple(dict.fromkeys(e.chunk_id for e in edges))
    refs = " ".join(f"[{c}]" for c in cites)
    text = f"{lead}: {listing(items)}. {detail}{refs}"
    rows = " ".join(dict.fromkeys(e.sentence for e in edges))
    tin = SYSTEM_TOKENS + 250 + count_tokens(question + rows)
    answer = Answer(
        text, cites, cites, False, False, tin, count_tokens(text)
    )
    return GraphAnswer(shape, list(items), answer)


def unique(xs):
    return list(dict.fromkeys(xs))


def has(g: Graph, plan: str, thing: str) -> list[Edge]:
    return g.find(plan, "includes", thing)


def query(g: Graph, question: str) -> GraphAnswer | None:
    """Turn a question into a graph query, if its shape fits.

    A real system asks the model to write the query (Cypher, SQL
    or a traversal) from the question and the schema. These
    patterns play that role for the shapes in the test set; any
    other question returns None and goes to search.
    """
    q = question.rstrip("?")
    m = re.search(r"not available on the (\w+) plan", q)
    if m and m.group(1) in PLANS:
        edges = g.find(m.group(1), "excludes")
        items = unique(e.target for e in edges)
        lead = f"Not available on {m.group(1)}"
        return respond(question, "excluded", lead, items, edges)
    m = re.search(r"available only on the (\w+) plan", q)
    if m and m.group(1) in PLANS:
        plan, edges, items = m.group(1), [], []
        for e in g.find(plan, "includes"):
            holders = {
                x.source
                for x in g.find(None, None, e.target)
                if x.relation in ("includes", "add-on")
            }
            if holders == {plan}:
                edges.append(e)
                items.append(e.target)
        lead = f"Only on {plan}"
        items = unique(items)
        return respond(question, "exclusive", lead, items, edges)
    m = re.fullmatch(r"Which plans (?:include|offer) (.+)", q)
    if m and ", and " not in m.group(1):
        wanted = things(m.group(1))
        plans = [
            p for p in PLANS if all(has(g, p, t) for t in wanted)
        ]
        edges = [
            e for p in plans for t in wanted for e in has(g, p, t)
        ]
        lead = f"Plans with {listing(wanted)}"
        return respond(question, "all-of", lead, plans, edges)
    m = re.fullmatch(r"Which releases changed (?:the )?(.+)", q)
    if m and link(m.group(1)):
        target = link(m.group(1))[0]
        edges = g.find(None, "changed", target)
        items = sorted(
            unique(f"release {e.source}" for e in edges)
        )
        lead = f"Releases that changed {target}"
        return respond(question, "changes", lead, items, edges)
    m = re.search(r"changed in releases (4\.\d) to (4\.\d)", q)
    if m:
        lo, hi = m.groups()
        edges = [
            e
            for e in g.find(None, "changed")
            if lo <= e.source <= hi and kind(e.target) != "plan"
        ]
        items = unique(f"{e.target} ({e.source})" for e in edges)
        lead = f"Changes in releases {lo} to {hi}"
        return respond(question, "global", lead, items, edges)
    m = re.search(
        r"does an? (\w+) workspace get that an? (\w+) workspace"
        r" does not",
        q,
    )
    if m and {m.group(1), m.group(2)} <= set(PLANS):
        hi, lo = m.groups()
        theirs = {
            e.target
            for e in g.find(lo)
            if e.relation in ("includes", "add-on")
        }
        edges = [
            e
            for e in g.find(hi, "includes")
            if e.target not in theirs
        ]
        items = unique(e.target for e in edges)
        lead = f"{hi} has, {lo} does not"
        return respond(question, "difference", lead, items, edges)
    m = re.search(
        r"connector that gained (.+) in release (4\.\d)", q
    )
    if m and q.startswith("Which plans"):
        gained, release = things(m.group(1)), m.group(2)
        hop = [
            e
            for e in g.find(release, "changed")
            if kind(e.target) == "connector"
        ]
        conns = [
            e.target
            for e in hop
            if any(g.find(release, "changed", t) for t in gained)
        ]
        edges = hop + [
            e for c in conns for e in g.find(None, "includes", c)
        ]
        plans = [
            p
            for p in PLANS
            if any(g.find(p, "includes", c) for c in conns)
        ]
        lead = f"Plans that can use {listing(unique(conns))}"
        return respond(question, "chain", lead, plans, edges)
    m = re.search(r"plans that include (.+?) commit to", q)
    if m and things(m.group(1)):
        feature = things(m.group(1))[0]
        plans = [p for p in PLANS if has(g, p, feature)]
        uptime = [
            e
            for p in plans
            for e in has(g, p, "uptime commitment")
        ]
        edges = [e for p in plans for e in has(g, p, feature)]
        detail = " ".join(unique(e.sentence for e in uptime))
        lead = f"Plans with {feature}"
        return respond(
            question,
            "chain",
            lead,
            plans,
            edges + uptime,
            detail + " " if detail else "",
        )
    m = re.search(r"error codes .* with the (\w+) connector", q)
    if m and m.group(1) in CONNECTORS:
        edges = [
            e
            for e in g.find(None, "related", m.group(1))
            if kind(e.source) == "error"
        ]
        items = unique(e.source for e in edges)
        lead = f"Seen with {m.group(1)}"
        return respond(question, "related", lead, items, edges)
    return None


EVERYONE = (
    User(  # what an unfiltered graph effectively answers as
        "everyone",
        frozenset(
            {PUBLIC, "customer", "staff", "support", "sales"}
            | {"finance", "eng", "hr"}
        ),
    )
)


class GraphAssistant:
    """Answer from the graph, search or a router between them.

    mode "graph": graph queries, and graph neighborhoods for every
    other question; "router": graph queries for the question
    shapes they fit, the search pipeline for everything else.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        graph: Graph,
        mode: str = "router",
        fresh: bool = True,
        permissions: bool = True,
    ):
        self.pipeline = pipeline
        self.graph = graph
        self.mode = mode
        self.fresh = fresh
        self.permissions = permissions

    def view(self, user: User) -> Graph:
        """The graph as this user may see it, right now."""
        who = user if self.permissions else EVERYONE
        docs = self.pipeline.by_id
        return self.graph.visible(who, docs, self.fresh)

    def ask(self, question: str, user: User | None = None):
        user = user or user_for("customer")
        g = self.view(user)
        found = query(g, question)
        if found is not None:
            return found  # an empty result abstains
        if self.mode == "router":
            result = self.pipeline.run(question, user)
            return GraphAnswer("", [], result.answer)
        return self.local(g, question)

    def local(self, g: Graph, question: str) -> GraphAnswer:
        """Answer from the chunks around the linked entities."""
        named = link(question)
        near = Counter(
            e.chunk_id
            for e in g.edges
            if e.source in named or e.target in named
        )
        top = [c for c, _ in near.most_common(4)]
        passages = [
            Passage(c, g.chunks[c].text, g.chunks[c].header)
            for c in top
        ]
        answer = generate(
            question, passages, MODELS["small"], GUARDED
        )
        return GraphAnswer("local", [], answer)


# Facts a careful reader states for the plan relations, used to
# check extraction: (plan, relation, feature or connector).
GOLD = {
    ("Business", "includes", "SSO"),
    ("Enterprise", "includes", "SSO"),
    ("Team", "add-on", "SSO"),
    ("Starter", "excludes", "SSO"),
    ("Enterprise", "includes", "SCIM provisioning"),
    ("Business", "includes", "audit logs"),
    ("Enterprise", "includes", "audit logs"),
    ("Starter", "excludes", "audit logs"),
    ("Team", "excludes", "audit logs"),
    ("Business", "includes", "larger workers"),
    ("Enterprise", "includes", "larger workers"),
    ("Team", "includes", "uptime commitment"),
    ("Business", "includes", "uptime commitment"),
    ("Enterprise", "includes", "uptime commitment"),
    ("Starter", "excludes", "uptime commitment"),
    ("Starter", "includes", "community forum"),
    ("Team", "includes", "support tickets"),
    ("Business", "includes", "support tickets"),
    ("Enterprise", "includes", "support tickets"),
    ("Enterprise", "includes", "technical account manager"),
    ("Business", "includes", "invoice billing"),
    ("Enterprise", "includes", "invoice billing"),
    ("Team", "includes", "Salesforce"),
    ("Business", "includes", "Salesforce"),
    ("Enterprise", "includes", "Salesforce"),
    ("Starter", "excludes", "Salesforce"),
    ("Team", "includes", "Stripe"),
    ("Business", "includes", "Stripe"),
    ("Enterprise", "includes", "Stripe"),
    ("Starter", "excludes", "Stripe"),
}


def check(graph: Graph) -> tuple[float, float, list[tuple]]:
    """Precision and recall of current plan facts against GOLD."""
    got = {
        (e.source, e.relation, e.target)
        for e in graph.edges
        if kind(e.source) == "plan"
    }
    right = got & GOLD
    precision = len(right) / len(got) if got else 0.0
    recall = len(right) / len(GOLD)
    return (
        round(precision, 3),
        round(recall, 3),
        sorted(got - GOLD),
    )


def assistant(mode: str = "router", **kw) -> GraphAssistant:
    pipe = Pipeline(PRODUCTION)
    full = Pipeline(PRODUCTION.but(drop_superseded=False))
    return GraphAssistant(pipe, build(full), mode, **kw)
