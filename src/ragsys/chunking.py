"""Split documents into chunks, the unit of retrieval.

Three strategies, from naive to structure-aware:
- fixed: windows of N words, blind to sentences and headings;
- sentence: whole sentences packed up to N words;
- section: one chunk per heading section, tables turned into
  self-contained rows, and a title header on every chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragsys.corpus import Document
from ragsys.text import sentences

STRATEGIES = ("fixed", "sentence", "section")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    header: str = ""

    @property
    def indexed_text(self) -> str:
        """What the index sees: header plus body."""
        return (
            f"{self.header}\n{self.text}"
            if self.header
            else self.text
        )


def fixed_chunks(
    doc: Document, size: int = 120, overlap: int = 0
) -> list[Chunk]:
    tokens = doc.text.split()
    step = max(size - overlap, 1)
    out = []
    for n, start in enumerate(range(0, len(tokens), step)):
        piece = " ".join(tokens[start : start + size])
        out.append(Chunk(f"{doc.id}#{n}", doc.id, piece))
        if start + size >= len(tokens):
            break
    return out


def pack(lines: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into groups of about size words."""
    groups: list[list[str]] = [[]]
    for line in lines:
        current = groups[-1]
        length = sum(len(s.split()) for s in current)
        if current and length + len(line.split()) > size:
            groups.append(current[-overlap:] if overlap else [])
        groups[-1].append(line)
    return ["\n".join(g) for g in groups if g]


def sentence_chunks(
    doc: Document, size: int = 120, overlap: int = 0
) -> list[Chunk]:
    groups = pack(sentences(doc.text), size, overlap)
    return [
        Chunk(f"{doc.id}#{n}", doc.id, text)
        for n, text in enumerate(groups)
    ]


def table_rows(lines: list[str]) -> list[str]:
    """Turn a Markdown table into one self-contained line each."""
    cells = [
        [c.strip() for c in line.strip("|").split("|")]
        for line in lines
        if not set(line) <= set("|- ")
    ]
    head, rows = cells[0], cells[1:]
    return [
        f"{head[0]} {row[0]}: "
        + "; ".join(
            f"{h} {v}"
            for h, v in zip(head[1:], row[1:], strict=False)
        )
        + "."
        for row in rows
    ]


def sections(text: str) -> list[tuple[str, list[str]]]:
    """Group lines under their nearest heading."""
    out: list[tuple[str, list[str]]] = [("", [])]
    table: list[str] = []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            table.append(line)
            continue
        if table:
            out[-1][1].extend(table_rows(table))
            table = []
        if line.startswith("#"):
            out.append((line.lstrip("# ").strip(), []))
        elif line.strip():
            out[-1][1].extend(sentences(line))
    return [(h, lines) for h, lines in out if lines]


def section_chunks(
    doc: Document, size: int = 160, overlap: int = 0
) -> list[Chunk]:
    out = []
    for heading, lines in sections(doc.text):
        header = doc.title
        if heading and heading != doc.title:
            header = f"{doc.title} > {heading}"
        for text in pack(lines, size, overlap):
            n = len(out)
            out.append(
                Chunk(f"{doc.id}#{n}", doc.id, text, header)
            )
    return out


def chunk_corpus(
    docs: list[Document],
    strategy: str = "section",
    size: int = 160,
    overlap: int = 0,
) -> list[Chunk]:
    """Chunk every document with one strategy."""
    fn = {
        "fixed": fixed_chunks,
        "sentence": sentence_chunks,
        "section": section_chunks,
    }[strategy]
    return [c for d in docs for c in fn(d, size, overlap)]
