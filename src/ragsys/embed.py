"""A simulated embedding model.

Real embedding models map text to vectors in which paraphrases
land close together. This stand-in does the same with a small,
explicit concept lexicon: "money back" and "refund" share a
concept, so their vectors are similar even with no word in
common. Exact codes such as LK-4012 carry little weight, which
is also true of many real models and is why hybrid search wins.
"""

from __future__ import annotations

import math
from collections import Counter

from ragsys.text import stem, tokenize, words

Vector = dict[str, float]

CONCEPTS: dict[str, tuple[str, ...]] = {
    "refund": ("refund", "money back", "reimburs", "repaid"),
    "cancel": (
        "cancel",
        "terminat",
        "close account",
        "quit",
        "end my plan",
        "stop paying",
    ),
    "retention": (
        "retention",
        "retain",
        "kept",
        "keep logs",
        "keep data",
        "store",
    ),
    "logs": ("log", "logs", "run history", "history"),
    "sso": (
        "sso",
        "single sign-on",
        "saml",
        "identity provider",
        "company login",
        "sign in",
    ),
    "price": (
        "price",
        "cost",
        "pricing",
        "fee",
        "fees",
        "pay",
        "paid",
        "payment",
        "how much",
        "$",
    ),
    "support": ("=support", "help", "ticket", "contact"),
    "response": ("response", "respond", "reply", "answer"),
    "uptime": (
        "uptime",
        "downtime",
        "outage",
        "availability",
        "down",
        "promised",
    ),
    "credit": ("credit", "compensat for downtime"),
    "export": ("export", "download", "backup"),
    "delete": ("delet", "remove", "erase", "purg"),
    "key": ("api key", "key", "keys", "token", "credential"),
    "rotate": ("rotat", "regenerat", "replace"),
    "region": (
        "region",
        "residency",
        "location",
        "eu-west",
        "us-east",
        "ap-south",
        "europe",
        "eu",
    ),
    "encrypt": ("encrypt", "aes-256", "tls"),
    "plan": ("plan", "tier", "subscription", "edition"),
    "pipeline": ("pipeline", "sync", "job", "jobs"),
    "connector": (
        "connector",
        "integration",
        "source",
        "destination",
    ),
    "limit": (
        "limit",
        "unlimited",
        "quota",
        "maximum",
        "allowance",
        "cap",
        "at most",
        "up to",
    ),
    "error": ("error", "fail", "broken", "stopp", "issue"),
    "remedy": (
        "fix",
        "resolv",
        "solv",
        "workaround",
        "persist",
        "raise",
        "reduc",
        "reauthoriz",
        "refresh",
    ),
    "cause": ("caus", "because", "due to", "reason", "why"),
    "auth": (
        "authenticat",
        "auth",
        "reauthoriz",
        "password",
        "oauth",
        "expired credentials",
    ),
    "rate": ("rate limit", "throttl", "too many request"),
    "schema": ("schema", "column", "drift", "evolution"),
    "memory": ("out of memory", "memory", "oom"),
    "discount": ("discount", "price reduction", "cheaper"),
    "oncall": ("on-call", "on call", "pager", "paged"),
    "stipend": ("stipend", "compensation", "bonus"),
    "severity": ("severity", "sev1", "sev2", "sev3", "priority"),
    "schedule": ("schedul", "interval", "frequency", "how often"),
    "audit": ("audit", "change history"),
    "role": (
        "role",
        "permission",
        "owner",
        "admin",
        "viewer",
        "editor",
    ),
    "trial": ("trial", "try", "evaluation period"),
    "users": ("users", "seats", "members", "people"),
    "backfill": ("backfill", "reload", "historical data"),
    "maintenance": ("maintenance", "planned work"),
    "scim": ("scim", "provision", "deprovision"),
    "security": ("soc", "security", "compliance", "vulnerabil"),
    "alert": ("alert", "notification", "notify", "slack"),
    "overage": (
        "overage",
        "exceed",
        "over the limit",
        "extra rows",
    ),
    "renewal": ("renew", "renewal"),
    "manager": (
        "account manager",
        "technical account manager",
        "tam",
        "csm",
    ),
    "speed": ("slow", "faster", "speed", "performance"),
    "annual": ("annual", "year", "yearly", "years"),
    "monthly": ("monthly", "month", "months"),
    "duration": (
        "day",
        "days",
        "hour",
        "hours",
        "week",
        "minute",
    ),
}

GENERIC = frozenset(
    "plan price duration support error limit pipeline monthly "
    "annual users".split()
)
CONCEPT_WEIGHT = 1.0
WORD_WEIGHT = 0.35


def _phrase_tokens(phrase: str) -> tuple[str, ...]:
    exact = phrase.startswith("=")
    toks = tuple(stem(w) for w in words(phrase)) or (phrase,)
    return ("=" + toks[0],) if exact else toks


_INDEX = {
    concept: [_phrase_tokens(p) for p in phrases]
    for concept, phrases in CONCEPTS.items()
}


def concepts(text: str) -> Counter[str]:
    """Count concept mentions, matching multi-word phrases."""
    raw = words(text) + (["$"] if "$" in text else [])
    toks = [stem(w) for w in raw]
    found: Counter[str] = Counter()
    for concept, phrases in _INDEX.items():
        for p in phrases:
            n = len(p)
            hits = sum(
                1
                for i in range(len(toks) - n + 1)
                if tuple(toks[i : i + n]) == p
                or (p[0][0] == "=" and raw[i] == p[0][1:])
                or (
                    n == 1
                    and len(p[0]) >= 4
                    and p[0][0] != "="
                    and toks[i].startswith(p[0])
                )
            )
            found[concept] += hits
    return +found


def embed(text: str) -> Vector:
    """Map text to a unit-length sparse vector."""
    vec: Vector = {}
    for c, n in concepts(text).items():
        weight = CONCEPT_WEIGHT * (0.5 if c in GENERIC else 1.0)
        vec[f"c:{c}"] = weight * (1 + math.log(n))
    for w, n in Counter(tokenize(text)).items():
        vec[f"w:{w}"] = WORD_WEIGHT * (1 + math.log(n))
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def cosine(a: Vector, b: Vector) -> float:
    """Dot product of two unit vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())
