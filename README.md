# Production RAG for AI Agents

Companion code for the book **Production RAG for AI Agents** by
Nachiketh Murthy.

Retrieval-augmented generation is easy to demo and hard to run.
This repository shows the whole path from a demo-quality pipeline
to one you can defend in production, and measures every step:

- chunking, lexical, dense and hybrid search, query expansion and
  reranking;
- context assembly, grounded answers with citations, abstention and
  verification;
- agents that decide when to search, chain searches across
  documents, combine records with documents and remember what users
  told them;
- permissions and tenants, freshness, planted instructions and
  personal data;
- latency, cost, approximate search and caching;
- a knowledge graph extracted from the documents, with provenance,
  and a router that sends relationship and aggregation questions
  to it and everything else to search;
- failure diagnosis and the improvement loop.

Everything runs offline. The knowledge base belongs to Larkspur
Cloud, a **fictional** data-pipeline company, and the embedding model,
reranker and language model are **simulated**, with the weaknesses
that matter in production built in: attention lost in the middle of
long contexts, confident answers from weak evidence, no sense of
which policy is newer unless told, and obedience to instructions
planted in documents. Change a setting and the measured results
change, reproducibly.

> The numbers come from a simulator and a 90-question test set, not
> from a production system. They show the methods and the shape of
> each trade-off; they are not benchmarks of any model or product.

## Quick start

```bash
git clone https://github.com/manifoldailearning/production-rag-for-agents
cd production-rag-for-agents
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # Python 3.11+, no runtime dependencies
pytest -q
python examples/ch01_demo_vs_production.py
```

Ask a question or grade a configuration from the command line:

```bash
ragsys ask "What does the Team plan cost per month?" --show
ragsys ask "Who is our technical account manager?" --tenant orchid
ragsys eval --config baseline
ragsys eval --config production --split test
```

## The configuration ladder

`ragsys/configs.py` defines each step from the baseline to the
production configuration. `examples/ch17_improvement_loop.py` prints
the table below on a fixed dev/test split.

| Step | Dev | Test | Leaks | Obeyed planted text |
|---|---|---|---|---|
| baseline | 57.8% | 50.0% | 9 | 8 |
| sections | 60.9% | 61.5% | 8 | 8 |
| hybrid | 62.5% | 65.4% | 8 | 10 |
| reranked | 60.9% | 65.4% | 9 | 13 |
| grounded | 71.9% | 76.9% | 8 | 0 |
| production | 78.1% | 84.6% | 0 | 0 |
| + multi-hop agent | 84.4% | 88.5% | 0 | 0 |

## Map of the repository

| Chapter | Module | Example |
|---|---|---|
| 1 Why RAG demos fail | `pipeline`, `configs` | `ch01_demo_vs_production.py` |
| 2 Measuring retrieval and answers | `evals` | `ch02_measuring.py` |
| 3 Documents and chunking | `corpus`, `chunking` | `ch03_chunking.py` |
| 4 Lexical, dense and hybrid search | `search`, `embed` | `ch04_search.py` |
| 5 Query rewriting and reranking | `rewrite`, `rerank` | `ch05_rerank_rewrite.py` |
| 6 Context assembly | `context` | `ch06_context.py` |
| 7 Grounded answers | `llm`, `grounding` | `ch07_grounding.py` |
| 8 When to search | `agent` | `ch08_when_to_search.py` |
| 9 Multi-hop retrieval | `agent` | `ch09_multihop.py` |
| 10 Records and documents | `structured` | `ch10_structured.py` |
| 11 Memory | `memory` | `ch11_memory.py` |
| 12 Access control and tenants | `access`, `cache` | `ch12_access.py` |
| 13 Freshness | `freshness` | `ch13_freshness.py` |
| 14 Planted instructions and personal data | `safety` | `ch14_safety.py` |
| 15 Latency, cost and caching | `cost`, `ann`, `cache` | `ch15_cost.py` |
| 16 GraphRAG: entities and graphs | `graph` | `ch16_graphrag_basics.py`, `ch16_graphs.py` |
| 17 Observability and the improvement loop | `trace`, `configs` | `ch17_improvement_loop.py` |

The data lives in `src/ragsys/data/`: 34 Markdown documents with
metadata headers; `questions.json`, the 90-question evaluation
set with gold documents, required facts and forbidden facts; and
`graph_questions.json`, ten aggregation and relationship questions
used in Chapter 16.

## When a graph pays off

`examples/ch16_graphrag_basics.py` is GraphRAG in miniature: seven
one-line documents, triples extracted from them, and a three-hop
traversal that answers "Who must be told if Postgres goes down?",
which keyword search cannot. It uses only the standard library.

`examples/ch16_graphs.py` extracts entities and relations from
every chunk, keeping the chunk each fact came from, and compares
search with a graph router:

| System | 10 graph questions | 90 original questions |
|---|---|---|
| search | 2/10 | 72/90 |
| multi-hop agent | 2/10 | |
| graph only | 9/10 | 29/90 |
| graph router | 9/10 | 72/90 |

The graph answers lists and relationships that search cannot, and
loses badly when used for everything. The router sends only the
question shapes a graph query fits, and the graph honors
permissions and freshness because every edge keeps its source.

## License

MIT. See `LICENSE`.
