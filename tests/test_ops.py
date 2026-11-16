from dataclasses import replace

from ragsys.access import user_for
from ragsys.ann import IVFIndex
from ragsys.cache import AnswerCache
from ragsys.chunking import chunk_corpus
from ragsys.configs import BASELINE, PRODUCTION
from ragsys.corpus import load_corpus
from ragsys.cost import estimate
from ragsys.evals import (
    evaluate,
    load_questions,
    split,
    summarize,
)
from ragsys.freshness import DocStore, IncrementalIndex
from ragsys.llm import Answer
from ragsys.pipeline import Pipeline
from ragsys.safety import clean, guard_output, quarantine, redact
from ragsys.trace import diagnose

DOCS = load_corpus()


def test_split_is_stable_and_disjoint():
    q = load_questions()
    dev, test = split(q, "dev"), split(q, "test")
    assert len(dev) + len(test) == len(q)
    assert not {x.id for x in dev} & {x.id for x in test}


def test_production_beats_baseline_without_leaks():
    questions = load_questions()
    base = summarize(evaluate(Pipeline(BASELINE), questions))
    prod = summarize(evaluate(Pipeline(PRODUCTION), questions))
    assert prod["accuracy"] > base["accuracy"] + 0.15
    assert prod["leaks"] == 0 and prod["injected"] == 0


def test_incremental_sync_embeds_only_changes():
    store = DocStore({d.id: d for d in DOCS})
    index = IncrementalIndex()
    index.sync(store)
    faq = store.docs["faq"]
    store.upsert(replace(faq, text=faq.text + "\nNew answer."))
    report = index.sync(store)
    assert report.changed == 1
    assert report.embedded_chunks == len(index.chunks["faq"])


def test_deletes_propagate_only_when_handled():
    store = DocStore({d.id: d for d in DOCS})
    lazy = IncrementalIndex(handle_deletes=False)
    lazy.sync(store)
    store.delete("faq")
    lazy.sync(store)
    assert "faq" in lazy.chunks


def test_quarantine_and_redaction():
    docs = {d.id: d for d in DOCS}
    tip = quarantine(docs["community-postgres-tip"])
    assert "SYSTEM NOTE" not in tip.text and "index" in tip.text
    ticket = redact(docs["support-ticket-sample"])
    assert "[email]" in ticket.text and "555" not in ticket.text
    _, findings = clean(DOCS)
    assert {f.kind for f in findings} == {"instruction", "pii"}


def test_output_guard_allows_own_domain():
    assert (
        guard_output("Write to security@larkspur.example")[1]
        == []
    )
    assert guard_output("Write to a@evil.example")[1]


def test_unscoped_cache_leaks_between_tenants():
    orchid = user_for("customer", "orchid")
    bramble = user_for("customer", "bramble")
    q = "How long are pipeline logs kept for our workspaces?"
    for scoped, leaks in ((False, True), (True, False)):
        cache = AnswerCache(scoped=scoped)
        cache.put(q, orchid, "v1", Answer("180 days"))
        assert (cache.get(q, bramble, "v1") is not None) is leaks


def test_semantic_cache_threshold():
    user = user_for("customer")
    a = "What does the Team plan cost per month?"
    b = "What does the Business plan cost per month?"
    loose = AnswerCache("semantic", 0.9)
    loose.put(a, user, "v1", Answer("$299"))
    assert loose.get(b, user, "v1") is not None
    strict = AnswerCache("semantic", 0.95)
    strict.put(a, user, "v1", Answer("$299"))
    assert strict.get(b, user, "v1") is None


def test_ivf_recall_rises_with_probes():
    chunks = chunk_corpus(DOCS, "section")
    ivf = IVFIndex(chunks, 6)
    q = "How long are logs kept on the Team plan?"
    one = {h.chunk.id for h in ivf.search(q, 4, 1)}
    every = {h.chunk.id for h in ivf.search(q, 4, 6)}
    assert len(every) == 4 and len(one) <= 4


def test_cost_scales_with_scanned_share():
    pipe = Pipeline(PRODUCTION)
    r = pipe.run("What is the Team plan price?")
    exact = estimate(r, PRODUCTION, 2_000_000)
    ivf = estimate(r, PRODUCTION, 2_000_000, scanned=0.05)
    assert exact.stages["vector"] == 20 * ivf.stages["vector"]


def test_diagnose_labels_stages():
    pipe = Pipeline(BASELINE)
    q = next(x for x in load_questions() if x.id == "q003")
    r = pipe.run(q.question)
    assert diagnose(q, r, pipe) == "generation"
