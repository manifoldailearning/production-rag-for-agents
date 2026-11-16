import importlib

import pytest

EXPECTED = {
    "ch01_demo_vs_production": "accuracy 55.6% on 90 questions",
    "ch02_measuring": "90 questions: 64 dev, 26 test",
    "ch03_chunking": "section",
    "ch04_search": "hybrid",
    "ch05_rerank_rewrite": "+ reranker",
    "ch06_context": "gold at position 4 of 8",
    "ch07_grounding": "+ verify",
    "ch08_when_to_search": "router",
    "ch09_multihop": "multi-hop:     6/6 correct",
    "ch10_structured": "records + docs: 9/9",
    "ch11_memory": "follow-ups, rewrite      7/7",
    "ch12_access": "pre-filter",
    "ch13_freshness": "drop superseded docs   8/8",
    "ch14_safety": "quarantine at ingest",
    "ch15_cost": "IVF 5%",
    "ch16_improvement_loop": "+ multi-hop",
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_runs(name, capsys):
    importlib.import_module(name).main()
    assert EXPECTED[name] in capsys.readouterr().out
