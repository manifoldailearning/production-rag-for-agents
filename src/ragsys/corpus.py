"""Load the knowledge base that every chapter retrieves from.

The corpus belongs to Larkspur Cloud, a fictional data-pipeline
company. Each document is a Markdown file with a small header of
metadata: last update, access groups, status and tenant. The
metadata matters as much as the text: it decides who may see a
document and whether it is still true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

PUBLIC = "public"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    type: str = "page"
    updated: str = "2026-01-01"
    acl: frozenset[str] = field(default=frozenset({PUBLIC}))
    status: str = "current"
    tenant: str = ""
    supersedes: str = ""

    @property
    def is_public(self) -> bool:
        return PUBLIC in self.acl and not self.tenant


def parse(raw: str) -> Document:
    """Parse one Markdown file with a --- header block."""
    _, header, body = raw.split("---\n", 2)
    meta = dict(
        line.split(": ", 1)
        for line in header.strip().splitlines()
    )
    acl = frozenset(g.strip() for g in meta.pop("acl").split(","))
    return Document(text=body.strip(), acl=acl, **meta)


def docs_dir() -> Path:
    return Path(str(resources.files("ragsys") / "data" / "docs"))


def load_corpus(path: Path | None = None) -> list[Document]:
    """All documents, sorted by id so runs are reproducible."""
    folder = path or docs_dir()
    files = sorted(folder.glob("*.md"))
    return [parse(f.read_text(encoding="utf-8")) for f in files]


def by_id(docs: list[Document]) -> dict[str, Document]:
    return {d.id: d for d in docs}
