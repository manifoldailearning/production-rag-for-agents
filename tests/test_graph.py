from ragsys.access import user_for
from ragsys.configs import PRODUCTION
from ragsys.evals import is_correct, load_questions
from ragsys.graph import assistant, build, check, query
from ragsys.pipeline import Pipeline

ROUTER = assistant("router")
CUSTOMER = user_for("customer")


def test_every_edge_keeps_its_chunk():
    g = ROUTER.graph
    assert g.edges
    assert all(e.chunk_id in g.chunks for e in g.edges)


def test_add_on_relation_fixes_team_sso():
    full = Pipeline(PRODUCTION.but(drop_superseded=False))
    one = build(full, 1).visible(CUSTOMER, full.by_id)
    two = build(full, 2).visible(CUSTOMER, full.by_id)
    assert ("Team", "includes", "SSO") in check(one)[2]
    assert ("Team", "includes", "SSO") not in check(two)[2]
    assert check(two)[1] == 1.0


def test_superseded_documents_leave_the_graph():
    q = "Which features are not available on the Team plan?"
    stale = assistant("router", fresh=False).ask(q)
    assert "SSO" in stale.items
    assert ROUTER.ask(q).items == ["audit logs"]


def test_permissions_apply_to_graph_answers():
    q = "Which error codes were seen with the Stripe connector?"
    open_graph = assistant("router", permissions=False)
    assert open_graph.ask(q, CUSTOMER).items == ["LK-4012"]
    filtered = ROUTER.ask(q, CUSTOMER)
    assert filtered.answer.abstained
    staff = ROUTER.ask(q, user_for("support"))
    assert staff.items == ["LK-4012"]


def test_router_leaves_original_questions_to_search():
    for q in load_questions():
        view = ROUTER.view(user_for(q.user, q.tenant))
        assert query(view, q.question) is None


def test_graph_questions_answered_by_router():
    qs = load_questions("graph_questions.json")
    right = sum(
        is_correct(q, a.text, a.abstained)
        for q in qs
        for a in [ROUTER.ask(q.question, CUSTOMER).answer]
    )
    assert right == 9
