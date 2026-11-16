from ragsys.chunking import chunk_corpus, table_rows
from ragsys.corpus import load_corpus
from ragsys.grounding import claims, verify
from ragsys.llm import MODELS, Passage, generate, position_weight
from ragsys.search import BM25Index, HybridIndex, VectorIndex, rrf
from ragsys.text import sentences, stem, tokenize


def test_stem_matches_word_forms():
    assert stem("cancelled") == stem("cancel")
    assert stem("released") == stem("release")
    assert stem("pipelines") == stem("pipeline")
    assert stem("lk-4012") == "lk-4012"


def test_tokenize_keeps_error_codes():
    assert "lk-4012" in tokenize("What does LK-4012 mean?")


def test_sentences_drop_table_separators():
    assert sentences("| a | b |\n|---|---|\nOne. Two.") == [
        "| a | b |",
        "One.",
        "Two.",
    ]


def test_corpus_metadata_is_parsed():
    docs = {d.id: d for d in load_corpus()}
    assert docs["pricing-v2"].status == "superseded"
    assert docs["tenant-orchid"].tenant == "orchid"
    assert "finance" in docs["finance-discounts"].acl


def test_table_rows_are_self_contained():
    rows = table_rows(
        ["| Plan | Price |", "|---|---|", "| Team | $1 |"]
    )
    assert rows == ["Plan Team: Price $1."]


def test_section_chunks_carry_headers():
    chunks = chunk_corpus(load_corpus(), "section")
    sso = [c for c in chunks if c.id == "pricing-v3#1"][0]
    assert sso.header == "Plans and Pricing > Single sign-on"


def test_bm25_finds_exact_codes():
    chunks = chunk_corpus(load_corpus(), "section")
    top = BM25Index(chunks).search("LK-4044", 1)[0]
    assert top.chunk.doc_id == "error-lk-4044"


def test_dense_finds_paraphrases():
    chunks = chunk_corpus(load_corpus(), "section")
    q = "I want to stop paying. How do I end my plan?"
    top = VectorIndex(chunks).search(q, 1)[0]
    assert top.chunk.doc_id == "cancel-subscription"


def test_rrf_rewards_agreement():
    chunks = chunk_corpus(load_corpus(), "section")
    hybrid = HybridIndex(chunks)
    fused = rrf(
        [
            hybrid.lexical.search("refund", 5),
            hybrid.dense.search("refund", 5),
        ],
        3,
    )
    assert len(fused) == 3
    assert fused[0].score >= fused[-1].score


def test_middle_of_context_gets_less_attention():
    assert position_weight(0, 8, 0.3) == 1.0
    assert position_weight(4, 8, 0.3) < 0.8


def test_abstains_on_weak_evidence():
    p = [Passage("x#0", "Pipelines run on a schedule.")]
    a = generate(
        "Is there a mobile app?",
        p,
        MODELS["small"],
        frozenset({"abstain"}),
    )
    assert a.abstained


def test_verify_drops_unsupported_sentences():
    p = [Passage("a#0", "Keys rotate in 24 hours.")]
    a = generate(
        "How long do keys keep working after rotation?",
        p,
        MODELS["small"],
        frozenset({"cite"}),
    )
    tampered = a.__class__(
        a.text + " Note: refunds are free.", a.used, a.citations
    )
    checked = verify(tampered, p)
    assert checked.dropped == ("Note: refunds are free.",)
    assert "24 hours" in checked.answer.text


def test_claims_read_citations():
    out = claims("One fact. [a#0] Another. [b#1]")
    assert [c.cited for c in out] == ["a#0", "b#1"]
