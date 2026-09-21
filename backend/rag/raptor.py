"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

The problem it solves: standard chunk-level retrieval answers questions
about ONE specific fact well ("what did you do at Coinbase?") but struggles
with questions that require synthesizing across MANY chunks at once
("summarize your overall career trajectory" or "what's your range across
companies?") — no single chunk contains that answer.

RAPTOR's approach: recursively cluster chunks by semantic similarity, then
have an LLM SUMMARIZE each cluster into a new, higher-level "summary node."
Repeat this clustering+summarizing on the summary nodes themselves, building
a tree — leaves are the original chunks, each higher layer is a broader
summary. At query time, both leaf chunks AND summary nodes are searchable,
so a broad synthesis question can match a high-level summary node directly,
while a specific-fact question still matches a precise leaf chunk.

Honest scope note: with only ~21 chunks in this corpus, we get a shallow
tree — one or two summary layers, not the many-layer trees RAPTOR shows on
large corpora. The mechanism is still real and correctly implemented; it's
just operating on a small input, which is exactly the size/complexity
mismatch discussed when deciding whether to use this here at all.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from anthropic import Anthropic
from sklearn.cluster import KMeans

from chunking import Chunk, load_corpus
from vector_store import VectorStore


@dataclass
class RaptorNode:
    id: str
    text: str
    level: int                      # 0 = original leaf chunk, 1+ = summary layers
    children: list[str] = field(default_factory=list)  # ids of nodes this summarizes
    source: str = "raptor_summary"


SUMMARY_PROMPT = """The following are related excerpts from a professional background \
corpus. Write a concise (3-5 sentence) summary that captures the common themes \
and key facts across all of them, suitable for answering broad, synthesis-style \
questions (e.g. "summarize your career" or "what's your overall experience with X").

Excerpts:
{excerpts}

Summary:"""


class RaptorBuilder:
    def __init__(self, vector_store: VectorStore, client: Anthropic | None = None,
                 model: str = "claude-sonnet-4-6", max_cluster_size: int = 4):
        self.vector_store = vector_store
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.max_cluster_size = max_cluster_size  # small on purpose, given this corpus's size

    def _summarize_cluster(self, texts: list[str]) -> str:
        excerpts = "\n\n---\n\n".join(texts)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=250,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(excerpts=excerpts)}],
        )
        return response.content[0].text.strip()

    def _cluster_level(self, nodes: list["RaptorNode"]) -> list[list["RaptorNode"]]:
        if len(nodes) <= self.max_cluster_size:
            return [nodes]  # small enough to summarize as one cluster

        embeddings = self.vector_store.embed([n.text for n in nodes])
        n_clusters = max(2, len(nodes) // self.max_cluster_size)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        clusters: dict[int, list[RaptorNode]] = {}
        for node, label in zip(nodes, labels):
            clusters.setdefault(int(label), []).append(node)
        return list(clusters.values())

    def build_tree(self, chunks: list[Chunk]) -> list["RaptorNode"]:
        """Returns ALL nodes across all levels — leaves plus every summary
        node built on top of them. All of these get embedded and indexed
        together at query time."""
        current_level: list[RaptorNode] = [
            RaptorNode(id=c.id, text=c.text, level=0, source=c.source) for c in chunks
        ]
        all_nodes: list[RaptorNode] = list(current_level)

        level = 1
        while len(current_level) > 1:
            clusters = self._cluster_level(current_level)
            next_level: list[RaptorNode] = []
            for i, cluster in enumerate(clusters):
                summary_text = self._summarize_cluster([n.text for n in cluster])
                summary_node = RaptorNode(
                    id=f"raptor_L{level}_{i}",
                    text=summary_text,
                    level=level,
                    children=[n.id for n in cluster],
                )
                next_level.append(summary_node)
            all_nodes.extend(next_level)

            if len(next_level) == len(current_level):
                break  # stop if clustering stops reducing node count (avoids infinite loop on tiny corpora)
            current_level = next_level
            level += 1

        return all_nodes


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python raptor.py <vector-bucket-name>")
        sys.exit(1)

    corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    chunks = load_corpus(corpus_dir)
    store = VectorStore(vector_bucket=sys.argv[1])
    store.build(chunks)

    builder = RaptorBuilder(store)
    tree = builder.build_tree(chunks)

    print(f"Built RAPTOR tree: {len(chunks)} leaf chunks -> {len(tree)} total nodes\n")
    for level in sorted(set(n.level for n in tree)):
        level_nodes = [n for n in tree if n.level == level]
        print(f"Level {level}: {len(level_nodes)} nodes")
        if level > 0:
            for n in level_nodes:
                print(f"  {n.id} (summarizes {len(n.children)} nodes): {n.text[:100]}...")