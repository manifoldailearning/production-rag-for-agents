"""Who is asking, and which documents they may read.

A retrieval system that ignores permissions is a search engine
for leaks. Every request carries a user with groups and, for
customers, a tenant. A document is readable when the user shares
one of its groups and, for tenant documents, is that tenant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragsys.chunking import Chunk
from ragsys.corpus import PUBLIC, Document


@dataclass(frozen=True)
class User:
    name: str
    groups: frozenset[str] = field(default=frozenset({PUBLIC}))
    tenant: str = ""


USERS = {
    "customer": User("customer", frozenset({PUBLIC, "customer"})),
    "sales": User("sales", frozenset({PUBLIC, "staff", "sales"})),
    "engineer": User(
        "engineer", frozenset({PUBLIC, "staff", "eng"})
    ),
    "support": User(
        "support", frozenset({PUBLIC, "staff", "support"})
    ),
}


def user_for(name: str, tenant: str = "") -> User:
    base = USERS[name]
    return User(base.name, base.groups, tenant)


def can_read(user: User, doc: Document) -> bool:
    """Group check first, then the tenant boundary."""
    if not user.groups & doc.acl:
        return False
    return not doc.tenant or doc.tenant == user.tenant


def readable_ids(user: User, docs: list[Document]) -> set[str]:
    return {d.id for d in docs if can_read(user, d)}


def filter_chunks(
    chunks: list[Chunk], allowed: set[str]
) -> list[Chunk]:
    return [c for c in chunks if c.doc_id in allowed]
