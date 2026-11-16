"""Named pipeline configurations, from naive to production.

Each step changes a few settings and is measured against the
question set. Keeping the ladder in code makes every number in
the book reproducible and every step easy to undo.
"""

from __future__ import annotations

from ragsys.pipeline import RagConfig

GROUNDED = frozenset({"cite", "abstain", "prefer_recent"})
GUARDED = GROUNDED | {"context_is_data"}

BASELINE = RagConfig()

LADDER = (
    BASELINE,
    BASELINE.but(
        name="sections", chunker="section", chunk_size=160
    ),
    BASELINE.but(
        name="hybrid",
        chunker="section",
        chunk_size=160,
        retriever="hybrid",
        expand_query=True,
    ),
    BASELINE.but(
        name="reranked",
        chunker="section",
        chunk_size=160,
        retriever="hybrid",
        expand_query=True,
        rerank=True,
    ),
    BASELINE.but(
        name="grounded",
        chunker="section",
        chunk_size=160,
        retriever="hybrid",
        expand_query=True,
        rerank=True,
        metadata=True,
        rules=GROUNDED,
        verify=True,
    ),
    BASELINE.but(
        name="production",
        chunker="section",
        chunk_size=160,
        retriever="hybrid",
        expand_query=True,
        rerank=True,
        metadata=True,
        rules=GUARDED,
        verify=True,
        acl=True,
        drop_superseded=True,
        tenant_first=True,
    ),
)

CONFIGS = {c.name: c for c in LADDER}
PRODUCTION = CONFIGS["production"]
