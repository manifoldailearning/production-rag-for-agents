"""Keep the index as current as the documents.

Documents change: prices are updated, policies are replaced,
pages are deleted. An index built once drifts away from the
truth. This module keeps a versioned document store and an index
that re-embeds only changed documents, removes deleted ones and
drops superseded versions, and counts the work each sync does.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

from ragsys.chunking import Chunk, chunk_corpus
from ragsys.corpus import Document
from ragsys.embed import Vector, embed
from ragsys.search import HybridIndex


def fingerprint(doc: Document) -> str:
    """Content hash: text plus metadata that changes answers."""
    raw = "|".join(
        [doc.text, doc.status, doc.updated, *sorted(doc.acl)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class DocStore:
    """The source of truth: current documents by id."""

    docs: dict[str, Document] = field(default_factory=dict)

    def upsert(self, doc: Document) -> None:
        self.docs[doc.id] = doc

    def delete(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)

    def supersede(self, old_id: str, new: Document) -> None:
        """Publish a new version; mark the old one superseded."""
        if old_id in self.docs:
            self.docs[old_id] = replace(
                self.docs[old_id], status="superseded"
            )
        self.upsert(replace(new, supersedes=old_id))


@dataclass
class SyncReport:
    added: int = 0
    changed: int = 0
    removed: int = 0
    embedded_chunks: int = 0


class IncrementalIndex:
    """Re-chunk and re-embed only what changed since last sync."""

    def __init__(
        self,
        drop_superseded: bool = True,
        handle_deletes: bool = True,
    ):
        self.drop_superseded = drop_superseded
        self.handle_deletes = handle_deletes
        self.seen: dict[str, str] = {}  # doc id -> fingerprint
        self.chunks: dict[str, list[Chunk]] = {}
        self.vectors: dict[str, Vector] = {}  # chunk id -> vector
        self.index: HybridIndex | None = None

    def wanted(self, doc: Document) -> bool:
        return not (
            self.drop_superseded and doc.status == "superseded"
        )

    def sync(self, store: DocStore) -> SyncReport:
        report = SyncReport()
        live = {
            i: d for i, d in store.docs.items() if self.wanted(d)
        }
        for doc_id, doc in live.items():
            fp = fingerprint(doc)
            if self.seen.get(doc_id) == fp:
                continue
            report.changed += doc_id in self.seen
            report.added += doc_id not in self.seen
            self.forget(doc_id)
            self.chunks[doc_id] = chunk_corpus([doc], "section")
            for c in self.chunks[doc_id]:
                self.vectors[c.id] = embed(c.indexed_text)
                report.embedded_chunks += 1
            self.seen[doc_id] = fp
        if self.handle_deletes:
            for doc_id in list(self.seen):
                if doc_id not in live:
                    self.forget(doc_id)
                    del self.seen[doc_id]
                    report.removed += 1
        self.index = HybridIndex(self.all_chunks(), self.vectors)
        return report

    def forget(self, doc_id: str) -> None:
        for c in self.chunks.pop(doc_id, []):
            self.vectors.pop(c.id, None)

    def all_chunks(self) -> list[Chunk]:
        return [c for cs in self.chunks.values() for c in cs]
