"""
Chunking for the twin's RAG corpus.

We chunk by markdown section (## headers), not fixed character windows —
each section in about.md / wpp-media-experience.md / etc. is already a
coherent unit of meaning (e.g. "Media Mix Modeling (MMM) — core expertise"),
so splitting on headers preserves context far better than blind character
splitting would for a corpus this structured and this small.

Two additional techniques layered on top of the base section-splitting,
both applied by build_index.py (not at Lambda cold start -- both need
either extra processing or an LLM call, so they belong in the offline
build step, not the request path):

1. Hierarchical (parent-child) chunking: large sections get split further
   into smaller "child" chunks for more precise embedding/matching, while
   each child keeps a reference to its parent's FULL text. Retrieval
   matches on the small, precise child; generation gets the fuller parent
   context. Small sections (the common case in this corpus) don't get
   split at all -- they're already precise enough to act as both parent
   and child.

2. Contextual retrieval (Anthropic's own published technique): before
   embedding, each chunk gets a short LLM-generated sentence of context
   prepended, describing where it fits in the broader document. This
   directly addresses a real weakness of pure structural chunking: a
   chunk retrieved in isolation can lose meaning if the context that
   established what it's about got cut off by the chunk boundary.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    id: str
    source: str          # filename, e.g. "wpp-media-experience.md"
    section: str         # the ## header this chunk came from
    text: str            # the actual chunk content (header + body) -- what gets embedded/matched
    metadata: dict = field(default_factory=dict)
    parent_text: str = ""  # the FULL parent section's text, for child chunks -- empty if this chunk IS the parent (no split needed)


def chunk_markdown_file(path: Path) -> list[Chunk]:
    """Split one markdown file into chunks, one per ## section.
    The top-level # title is prepended to every chunk as context."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    title = ""
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()

    # Split on '## ' section headers
    section_pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(section_pattern.finditer(raw))

    chunks = []

    # Any text between the # title and the FIRST ## header was previously
    # silently dropped entirely -- the loop below only ever looked at spans
    # BETWEEN matches, never the text before the first one. Confirmed as a
    # real bug losing content in 2 of 4 corpus files (an intro/elevator-
    # pitch sentence right after each title). Captured here as its own
    # "Introduction" chunk when present.
    intro_end = matches[0].start() if matches else len(raw)
    intro_start = len(lines[0]) + 1 if title else 0  # skip past the title line itself
    intro_text = raw[intro_start:intro_end].strip()
    if intro_text:
        text = f"{title} — Introduction\n\n{intro_text}" if title else intro_text
        chunks.append(Chunk(
            id=f"{path.stem}::intro",
            source=path.name,
            section="Introduction",
            text=text,
            metadata={"doc_title": title},
        ))

    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()

        # Prepend title + section header for context — this matters a lot
        # for a corpus this small, since a chunk retrieved in isolation
        # should still make sense without the rest of the document.
        text = f"{title} — {section_name}\n\n{body}" if title else f"{section_name}\n\n{body}"

        chunks.append(Chunk(
            id=f"{path.stem}::{i}",
            source=path.name,
            section=section_name,
            text=text,
            metadata={"doc_title": title},
        ))

    # Fallback: if a file has no ## sections at all, treat the whole file as one chunk
    if not chunks and raw.strip():
        chunks.append(Chunk(
            id=f"{path.stem}::0",
            source=path.name,
            section=title or path.stem,
            text=raw.strip(),
            metadata={"doc_title": title},
        ))

    return chunks


def split_into_children(chunk: Chunk, max_child_chars: int = 500) -> list[Chunk]:
    """Hierarchical (parent-child) split. If a section is already small
    (the common case in this corpus -- most sections run 200-1000 chars),
    it's returned unchanged: the parent chunk itself is precise enough to
    be its own child, no split needed. Larger sections get split on
    paragraph boundaries into smaller children, each carrying the FULL
    parent text in parent_text -- so a search match on a precise child
    still gives the generator the complete surrounding context."""
    if len(chunk.text) <= max_child_chars:
        return [chunk]

    # chunk.text is "{title} — {section}\n\n{body}" (see chunk_markdown_file).
    # Split only the BODY into paragraphs -- splitting the whole text would
    # treat the header line itself as its own tiny, near-useless paragraph,
    # separate from the real content.
    header, _, body = chunk.text.partition("\n\n")
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        return [chunk]  # nothing meaningful to split on

    children = []
    for i, para in enumerate(paragraphs):
        children.append(Chunk(
            id=f"{chunk.id}::child{i}",
            source=chunk.source,
            section=chunk.section,
            text=f"{header}\n\n{para}",  # each child keeps the header for standalone context, matching the original construction
            metadata=chunk.metadata,
            parent_text=chunk.text,  # the full original section, for generation-time context
        ))
    return children


CONTEXT_PROMPT = """You are helping prepare a document chunk for retrieval. Given \
the full document and one specific chunk from it, write ONE short sentence (max 20 \
words) situating this chunk within the broader document -- what it's part of, so \
the chunk makes sense even when read in isolation from the rest of the document. \
Return ONLY that sentence, nothing else.

Full document:
{document}

Chunk to contextualize:
{chunk_text}

Context sentence:"""


def contextualize_chunks(chunks: list[Chunk], full_document_text: str, client) -> list[Chunk]:
    """Anthropic's own published Contextual Retrieval technique: prepends a
    short LLM-generated context sentence to each chunk before embedding.
    Makes a real LLM call per chunk -- this is why it's an offline
    build_index.py step, not something run at Lambda cold start or per
    request. Returns NEW Chunk objects; the original chunks list is
    untouched."""
    contextualized = []
    for chunk in chunks:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=80,
            messages=[{"role": "user", "content": CONTEXT_PROMPT.format(
                document=full_document_text, chunk_text=chunk.text,
            )}],
        )
        context_sentence = response.content[0].text.strip()
        contextualized.append(Chunk(
            id=chunk.id,
            source=chunk.source,
            section=chunk.section,
            text=f"{context_sentence}\n\n{chunk.text}",
            metadata=chunk.metadata,
            parent_text=chunk.parent_text,
        ))
    return contextualized


def load_corpus(corpus_dir: Path) -> list[Chunk]:
    """Load and chunk every .md file in the corpus directory. Applies
    hierarchical splitting (split_into_children) but NOT contextual
    retrieval -- that needs an LLM client and is applied separately by
    build_index.py, keeping this function fast and dependency-free (no
    API key needed just to load and chunk the corpus, e.g. for testing)."""
    all_chunks = []
    for md_file in sorted(corpus_dir.glob("*.md")):
        section_chunks = chunk_markdown_file(md_file)
        for chunk in section_chunks:
            all_chunks.extend(split_into_children(chunk))
    return all_chunks


if __name__ == "__main__":
    corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    chunks = load_corpus(corpus_dir)
    print(f"Loaded {len(chunks)} chunks from {corpus_dir}\n")
    for c in chunks:
        is_child = " (child)" if c.parent_text else ""
        print(f"[{c.id}] ({c.source} — {c.section}){is_child} — {len(c.text)} chars")