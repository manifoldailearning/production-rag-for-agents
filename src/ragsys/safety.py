"""Treat retrieved content as untrusted input.

Anything that can be indexed can reach the model: a forum post,
a support ticket, a web page. Defences work best in layers:
- at ingestion, quarantine text that addresses the model and
  redact personal data that no answer needs;
- in the prompt, tell the model that context is data;
- after generation, verify answers against the context and
  block output that contains contact details not on an allowlist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ragsys.corpus import Document
from ragsys.llm import INJECTION
from ragsys.text import sentences

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"\+?\d[\d ()-]{8,}\d")
ALLOWED_DOMAINS = ("larkspur.example",)


@dataclass(frozen=True)
class Finding:
    doc_id: str
    kind: str  # "instruction" or "pii"
    text: str


def scan(doc: Document) -> list[Finding]:
    """Find instructions aimed at a model, and personal data."""
    out = []
    lines = sentences(doc.text)
    for i, line in enumerate(lines):
        if INJECTION.search(line):
            payload = " ".join(lines[i : i + 2])
            out.append(Finding(doc.id, "instruction", payload))
    for pattern in (EMAIL, PHONE):
        for m in pattern.finditer(doc.text):
            if not m.group(0).endswith(ALLOWED_DOMAINS):
                out.append(Finding(doc.id, "pii", m.group(0)))
    return out


def quarantine(doc: Document) -> Document:
    """Remove sentences addressed to the model, and the next."""
    lines = sentences(doc.text)
    drop: set[int] = set()
    for i, line in enumerate(lines):
        if INJECTION.search(line):
            drop |= {i, i + 1}
    text = doc.text
    for i in sorted(drop):
        if i < len(lines):
            text = text.replace(lines[i], "")
    text = re.sub(r"\n\s*\n(\s*\n)+", "\n\n", text).strip()
    return replace(doc, text=text)


def redact(doc: Document) -> Document:
    """Replace emails and phone numbers outside the allowlist."""

    def mask(m: re.Match) -> str:
        found = m.group(0)
        if found.endswith(ALLOWED_DOMAINS):
            return found
        return "[email]" if "@" in found else "[phone]"

    text = PHONE.sub(mask, EMAIL.sub(mask, doc.text))
    return replace(doc, text=text)


def clean(
    docs: list[Document],
) -> tuple[list[Document], list[Finding]]:
    """Ingestion-time defences for a whole corpus."""
    findings = [f for d in docs for f in scan(d)]
    return [redact(quarantine(d)) for d in docs], findings


def guard_output(text: str) -> tuple[str, list[str]]:
    """Block answers that hand out unapproved contact details."""
    blocked = [
        m.group(0)
        for m in EMAIL.finditer(text)
        if not m.group(0).endswith(ALLOWED_DOMAINS)
    ]
    if blocked:
        return (
            "I can't share that. Please use the support portal.",
            blocked,
        )
    return text, []
