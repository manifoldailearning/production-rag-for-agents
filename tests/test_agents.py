from ragsys.access import user_for
from ragsys.agent import (
    MultiHopAgent,
    closed_book,
    decompose,
    needs_retrieval,
)
from ragsys.configs import PRODUCTION
from ragsys.memory import MemoryStore, contextualize, personalize
from ragsys.pipeline import Pipeline
from ragsys.structured import RecordAssistant, find_account

PIPE = Pipeline(PRODUCTION)


def test_router_skips_definitions_only():
    assert not needs_retrieval("What is exponential backoff?")
    assert needs_retrieval("What is the Team plan price?")
    assert needs_retrieval("Which destinations are supported?")


def test_closed_book_answers_from_stale_training():
    a = closed_book(
        "Within how many days can annual plans be refunded?"
    )
    assert "14 days" in a.text


def test_decompose_chained_question():
    plan = decompose(
        "For the plan you get during the free trial, what is the "
        "log retention?"
    )
    assert plan.kind == "plan"
    assert plan.then == "For {x}, what is the log retention?"


def test_multihop_finds_bridge_entity():
    q = (
        "In the release that added SCIM provisioning, which new "
        "region became available?"
    )
    result = MultiHopAgent(PIPE).run(q)
    assert "ap-south" in result.answer.text
    assert result.hops[1].query.startswith("In release 4.6")


def test_record_tool_requires_exact_name():
    assert (
        find_account("calder media's plan").name == "Calder Media"
    )
    assert find_account("Calder's plan") is None


def test_customers_cannot_read_other_accounts():
    bot = RecordAssistant(PIPE)
    reply = bot.ask(
        "What plan is Calder Media on?",
        user_for("customer", "orchid"),
    )
    assert reply.route == "denied"


def test_follow_up_is_rewritten():
    q = contextualize(
        "What about Business?",
        ["How long are logs kept on the Team plan?"],
    )
    assert q == "How long are logs kept on the Business plan?"


def test_slot_memory_keeps_latest_fact():
    store = MemoryStore("slots")
    store.observe("u", "We're on the Team plan.", 0)
    store.observe("u", "We upgraded to the Business plan.", 1)
    assert store.recall("u", "plan") == "Business"
    store.observe("u", "Please forget my plan.", 2)
    assert store.recall("u", "plan") is None


def test_append_memory_returns_stale_fact():
    store = MemoryStore("append")
    store.observe("u", "We're on the Team plan.", 0)
    store.observe("u", "We upgraded to the Business plan.", 1)
    assert store.recall("u", "plan") == "Team"


def test_personalize_only_personal_questions():
    assert personalize(
        "How long are my logs kept?", "Team"
    ).endswith("on the Team plan?")
    assert personalize("What is SCIM?", "Team") == "What is SCIM?"
