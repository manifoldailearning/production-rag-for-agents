"""Check that an answer is supported by the context it cites.

A citation is a claim: "this sentence came from that passage".
Verification tests the claim. Each answer sentence must appear
in the passage it cites, or in some passage when citations are
off. Unsupported sentences are removed; if nothing survives, the
system abstains instead of shipping an unsupported answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ragsys.llm import ABSTAIN, Answer, Passage
from ragsys.text import count_tokens, sentences

CITE = re.compile(r"\s*\[([^\]]+)\]")


@dataclass(frozen=True)
class Claim:
    sentence: str
    cited: str  # chunk id, or "" when the answer has no citations


def claims(text: str) -> list[Claim]:
    """Split an answer into sentences and their citations."""
    out = []
    for part in re.split(
        r"(?<=\])\s+|(?<=[.!?])\s+(?=[A-Z])", text
    ):
        part = part.strip()
        if not part:
            continue
        m = CITE.search(part)
        sentence = CITE.sub("", part).strip()
        out.append(Claim(sentence, m.group(1) if m else ""))
    return out


def supported(claim: Claim, passages: list[Passage]) -> bool:
    pool = [p for p in passages if p.chunk_id == claim.cited]
    if not claim.cited:
        pool = passages
    return any(claim.sentence in sentences(p.text) for p in pool)


@dataclass(frozen=True)
class Verdict:
    answer: Answer
    kept: int
    dropped: tuple[str, ...]


def verify(answer: Answer, passages: list[Passage]) -> Verdict:
    """Keep only supported sentences; abstain if none remain."""
    if answer.abstained:
        return Verdict(answer, 0, ())
    kept, dropped = [], []
    for c in claims(answer.text):
        if supported(c, passages):
            kept.append(
                f"{c.sentence} [{c.cited}]"
                if c.cited
                else c.sentence
            )
        else:
            dropped.append(c.sentence)
    if not kept:
        safe = replace(
            answer,
            text=ABSTAIN,
            used=(),
            citations=(),
            abstained=True,
            injected=False,
            tokens_out=count_tokens(ABSTAIN),
        )
        return Verdict(safe, 0, tuple(dropped))
    text = " ".join(kept)
    checked = replace(
        answer,
        text=text,
        injected=answer.injected and not dropped,
        tokens_out=count_tokens(text),
    )
    return Verdict(checked, len(kept), tuple(dropped))
