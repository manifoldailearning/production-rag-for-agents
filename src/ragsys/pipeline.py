"""The retrieval-augmented generation pipeline, end to end.

One configuration object describes every choice the book
discusses: chunking, index type, reranking, context assembly,
permissions, metadata and the prompt rules given to the model.
Changing the configuration and re-running the evaluation is how
every improvement in the book is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ragsys.access import User, readable_ids, user_for
from ragsys.chunking import Chunk, chunk_corpus
from ragsys.context import assemble
from ragsys.corpus import Document, load_corpus
from ragsys.grounding import verify
from ragsys.llm import MODELS, Answer, Passage, generate
from ragsys.rerank import rerank
from ragsys.rewrite import expand
from ragsys.search import Hit, build_index


@dataclass(frozen=True)
class RagConfig:
    name: str = "baseline"
    chunker: str = "fixed"
    chunk_size: int = 120
    overlap: int = 0
    retriever: str = "bm25"
    expand_query: bool = False
    candidates: int = 20
    rerank: bool = False
    top_k: int = 4
    budget: int = 1500
    order: str = "rank"
    dedupe: bool = False
    compress_to: int = 0
    metadata: bool = False
    acl: bool = False
    acl_mode: str = "pre"  # filter before ranking, or "post"
    drop_superseded: bool = False
    tenant_first: bool = False  # a tenant's own terms rank first
    model: str = "small"
    rules: frozenset[str] = field(default_factory=frozenset)
    verify: bool = False

    def but(self, **changes) -> RagConfig:
        """A copy with some settings changed."""
        return replace(self, **changes)


@dataclass
class Result:
    question: str
    user: User
    query: str
    hits: list[Hit]
    context: list[Chunk]
    answer: Answer
    stages: dict[str, float] = field(default_factory=dict)


class Pipeline:
    """Index a corpus once; answer questions with one config."""

    def __init__(
        self,
        config: RagConfig,
        docs: list[Document] | None = None,
        index=None,
    ):
        self.config = config
        docs = load_corpus() if docs is None else docs
        self.by_id = {d.id: d for d in docs}
        if config.drop_superseded:
            docs = [d for d in docs if d.status != "superseded"]
        self.docs = docs
        if (
            index is not None
        ):  # built elsewhere, e.g. incrementally
            self.chunks, self.index = index.chunks, index
            return
        self.chunks = chunk_corpus(
            docs,
            config.chunker,
            config.chunk_size,
            config.overlap,
        )
        self.index = build_index(self.chunks, config.retriever)

    def passage(self, chunk: Chunk) -> Passage:
        """What the model reads for one chunk."""
        if not self.config.metadata:
            return Passage(chunk.id, chunk.text, chunk.header)
        doc = self.by_id[chunk.doc_id]
        label = f"{chunk.header or doc.title} ({doc.status}, "
        label += f"updated {doc.updated})"
        return Passage(
            chunk.id, chunk.text, label, doc.updated, doc.status
        )

    def retrieve(self, question: str, user: User):
        cfg = self.config
        allowed = (
            readable_ids(user, self.docs) if cfg.acl else None
        )
        query = expand(question) if cfg.expand_query else question
        pre = allowed if cfg.acl_mode == "pre" else None
        hits = self.index.search(query, cfg.candidates, pre)
        if allowed is not None and cfg.acl_mode == "post":
            hits = [h for h in hits if h.chunk.doc_id in allowed]
        if cfg.rerank:
            hits = rerank(question, hits, cfg.candidates)
        if cfg.tenant_first and user.tenant:
            mine = [
                h
                for h in hits
                if self.tenant_of(h) == user.tenant
            ]
            hits = mine + [h for h in hits if h not in mine]
        return query, hits

    def tenant_of(self, hit: Hit) -> str:
        return self.by_id[hit.chunk.doc_id].tenant

    def run(
        self, question: str, user: User | None = None
    ) -> Result:
        cfg = self.config
        user = user or user_for("customer")
        query, hits = self.retrieve(question, user)
        context = assemble(
            question,
            [h.chunk for h in hits],
            cfg.top_k,
            cfg.budget,
            cfg.order,
            cfg.dedupe,
            cfg.compress_to,
        )
        passages = [self.passage(c) for c in context]
        answer = generate(
            question, passages, MODELS[cfg.model], cfg.rules
        )
        if cfg.verify:
            answer = verify(answer, passages).answer
        stages = {
            "candidates": len(hits),
            "reranked": len(hits) if cfg.rerank else 0,
            "context": len(context),
        }
        return Result(
            question, user, query, hits, context, answer, stages
        )
