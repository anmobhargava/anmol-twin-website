"""
Offline indexing pipeline. Run this manually whenever corpus/ changes
(and, later, by a CI/CD job on the same trigger) -- NOT run by Lambda at
request time or at cold start. This is the one place all the "build-time"
techniques (contextual retrieval, RAPTOR) actually execute; Lambda only
ever queries what this script uploads to S3 Vectors.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python build_index.py <vector-bucket-name>

Requires AWS credentials configured (for Bedrock + S3 Vectors) and an
Anthropic API key (for RAPTOR summarization and contextual retrieval).
"""

import os
import sys
from pathlib import Path

from anthropic import Anthropic

# rag/ modules use simple (non-package) imports internally (e.g. `from chunking
# import Chunk`), matching the same pattern as pipeline.py -- so this needs
# rag/ added to sys.path directly, not imported as a package (e.g. NOT
# `from rag.raptor import RaptorBuilder`, which would break raptor.py's own
# internal `from chunking import Chunk` the moment Python treats rag/ as a
# package rather than a plain directory on the path).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from rag.chunking import Chunk, contextualize_chunks, load_corpus
from rag.raptor import RaptorBuilder
from rag.vector_store import VectorStore

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def build_and_upload_index(vector_bucket: str):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    print(f"Loading corpus from {CORPUS_DIR}...")
    leaf_chunks = load_corpus(CORPUS_DIR)
    print(f"Loaded {len(leaf_chunks)} leaf chunks (after hierarchical splitting where applicable).")

    # Contextual retrieval: each chunk needs the FULL document it came from
    # to generate a meaningful context sentence -- grouping by source file
    # here so contextualize_chunks() gets the right document per chunk,
    # not the whole corpus concatenated together.
    print("Applying contextual retrieval (this makes one LLM call per chunk)...")
    contextualized_chunks: list[Chunk] = []
    by_source: dict[str, list[Chunk]] = {}
    for chunk in leaf_chunks:
        by_source.setdefault(chunk.source, []).append(chunk)

    for source_file, chunks_in_file in by_source.items():
        full_doc_text = (CORPUS_DIR / source_file).read_text(encoding="utf-8")
        contextualized_chunks.extend(contextualize_chunks(chunks_in_file, full_doc_text, client))
    print(f"Contextualized {len(contextualized_chunks)} chunks.")

    # RAPTOR needs a VectorStore to compute embeddings for clustering --
    # this is a real, temporary use of the vector store purely for its
    # embed() method, not for building the final searchable index yet.
    print("Building RAPTOR tree (clustering + LLM summarization)...")
    temp_store = VectorStore(vector_bucket=vector_bucket, index_name="raptor-scratch")
    raptor_builder = RaptorBuilder(temp_store, client=client)
    tree = raptor_builder.build_tree(contextualized_chunks)

    all_chunks: list[Chunk] = list(contextualized_chunks)
    for node in tree:
        if node.level > 0:
            all_chunks.append(Chunk(
                id=node.id,
                source="raptor_summary",
                section=f"Synthesis summary (level {node.level})",
                text=node.text,
                metadata={"raptor_level": node.level},  # note: "children" (a list) is dropped here -- S3 Vectors metadata must be simple types, see vector_store.py's build() comment
            ))
    print(f"Total nodes after folding in RAPTOR summaries: {len(all_chunks)}")

    print(f"Uploading {len(all_chunks)} vectors to S3 Vectors (bucket={vector_bucket}, index=corpus)...")
    final_store = VectorStore(vector_bucket=vector_bucket, index_name="corpus")
    final_store.build(all_chunks)
    print("Done. Lambda's next cold start will query this index directly -- no rebuild needed on its end.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_index.py <vector-bucket-name>")
        sys.exit(1)
    build_and_upload_index(vector_bucket=sys.argv[1])