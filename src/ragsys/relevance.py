"""Joint question-passage relevance, used by simulated models.

A cross-encoder reranker and a generator both read the question
and the passage together, which lets them notice what a vector
comparison misses: whether the passage mentions the specific
plan, code or version the question is about. This scorer plays
that role for the simulated reranker and the simulated model.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ragsys.embed import GENERIC, concepts
from ragsys.text import STOPWORDS, stem, tokenize, words

_CAPITAL = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:-[A-Z0-9]+)?\b")
_CODE = re.compile(r"\b[A-Za-z]*-?\d[\w.-]*\b")
IGNORE = {"larkspur", "cloud", "i", "can", "does", "is", "how"}


@lru_cache(maxsize=50_000)
def entities(question: str) -> frozenset[str]:
    """Names and codes the answer must be about."""
    found = set()
    first = question.split()[0] if question.split() else ""
    for m in _CAPITAL.finditer(question):
        word = m.group(0)
        if word != first and word.lower() not in IGNORE:
            found.add(stem(word.lower()))
    for m in _CODE.finditer(question):
        found.add(m.group(0).lower().rstrip(".?"))
    return frozenset(e for e in found if e not in STOPWORDS)


NOISE = {"larkspur", "cloud"}
SHAPES = [  # question pattern -> what an answer looks like
    (
        r"^how (long|quickly|soon|often)",
        r"\d+\s?-?(business )?(day|hour|minute|week|month|year)"
        r"|one year|next business day|within",
    ),
    (r"^how many", r"\d"),
    (
        r"^how much|what does .* cost|\bcost\b|\bprice\b",
        r"\$|\bcustom\b",
    ),
    (
        r"^when\b",
        r"\b\d{1,2} [A-Z][a-z]+ 20\d\d|\b20\d\d\b|sundays?\b"
        r"|end of",
    ),
    (
        r"^(what caused|what causes|why)",
        r"cause|because|due to|means",
    ),
    (r"^who\b", r"[a-z] [A-Z][a-z]+ [A-Z][a-z]+"),
]
PROCEDURE = re.compile(
    r"^(how (do|can|should) (i|we)|what should (i|we) do)"
    r"|\bfix\b",
    re.I,
)
ACTIONS = set(
    """open opens choose select upload uploads enter request
    raise reduce check enable turn add create generate cancel
    rotate revoke export contact include apply update refresh
    reauthorize""".split()
)


def content(text: str) -> set[str]:
    return set(tokenize(text)) - NOISE


def entity_share(question: str, text: str) -> float:
    """Share of the question's names and codes found in text."""
    ents = entities(question)
    if not ents:
        return 1.0
    present = set(words(text)) | set(tokenize(text))
    return sum(1 for e in ents if e in present) / len(ents)


def weight(found: set[str]) -> float:
    """Generic concepts such as plan or price count half."""
    return sum(0.5 if c in GENERIC else 1.0 for c in found)


def coverage(
    question: str, text: str, concepts_too: bool = True
) -> float:
    """Share of the question's words (and concepts) in text.

    Counting concepts as matches means a paraphrase is covered.
    """
    want, have = content(question), content(text)
    if concepts_too:
        want |= {f"c:{c}" for c in concepts(question)}
        have |= {f"c:{c}" for c in concepts(text)}
    return len(want & have) / len(want) if want else 0.0


@lru_cache(maxsize=200_000)
def relevance(
    question: str, passage: str, miss_penalty: float = 0.5
) -> float:
    """Score in [0, 1], mostly coverage of the question.

    A small precision term prefers focused passages. A passage
    that never mentions a named entity from the question is
    multiplied down by miss_penalty.
    """
    qc, pc = set(concepts(question)), set(concepts(passage))
    word = coverage(question, passage, concepts_too=False)
    conc = weight(qc & pc) / weight(qc) if qc else word
    focus = weight(qc & pc) / weight(pc) if pc else 0.0
    score = 0.45 * conc + 0.45 * word + 0.1 * focus
    share = entity_share(question, passage)
    score *= miss_penalty + (1 - miss_penalty) * share
    return round(score, 4)


def answer_fit(question: str, sentence: str) -> float:
    """Does the sentence have the shape of the answer asked for?

    Durations for how-long questions, amounts for how-much,
    steps for how-do-I, causes for why and names for who.
    Returns 1.15 for a match, 0.75 for a miss, 1.0 otherwise.
    """
    q = question.strip()
    for ask, shape in SHAPES:
        if re.search(ask, q, re.I):
            hit = re.search(shape, sentence, re.I)
            if ask.startswith("^who"):
                hit = re.search(shape, sentence)
            return 1.15 if hit else 0.75
    if PROCEDURE.search(q):
        return 1.15 if ACTIONS & set(words(sentence)) else 0.8
    return 1.0
