"""
Hybrid retrieval: combines BM25 (sparse, keyword-based) with dense vector
search (semantic), fused via Reciprocal Rank Fusion (RRF).

Why hybrid at all: dense embeddings are great at semantic similarity
("MMM work" matches "Media Mix Modeling" even without shared words) but
can miss exact-term matches that BM25 catches trivially (e.g. a query for
"WPP Media" should strongly favor chunks that literally say "WPP Media").
Combining both catches cases either one alone would miss.

RRF, not a weighted average: RRF combines RANKS (position 1, 2, 3...) from
each retriever rather than raw scores. This sidesteps the problem that BM25
scores and cosine-similarity scores live on completely different, incomparable
scales — you can't meaningfully average a BM25 score of 8.3 with a cosine
similarity of 0.71. Ranks are always comparable regardless of the underlying
scoring method.
"""

from pathlib import Path

from rank_bm25 import BM25Okapi

from chunking import Chunk, load_corpus
from vector_store import VectorStore


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], vector_store: VectorStore):
        self.chunks = chunks
        self.vector_store = vector_store
        # BM25 needs pre-tokenized text — simple whitespace/lowercase split
        # is sufficient for a corpus this size and domain
        tokenized = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_ranked_ids(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i].id for i in ranked_indices]

    def _dense_ranked_ids(self, query: str, k: int) -> list[str]:
        results = self.vector_store.search(query, k=k)
        return [chunk.id for chunk, _ in results]

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> list[tuple[Chunk, float]]:
        """rrf_k is RRF's smoothing constant — 60 is the standard default from
        the original RRF paper, dampening the impact of any single retriever's
        top rank so no one method dominates the fused result."""
        candidate_pool = max(k * 3, 10)  # retrieve more from each method than we need, to give fusion real signal to work with
        bm25_ids = self._bm25_ranked_ids(query, candidate_pool)
        dense_ids = self._dense_ranked_ids(query, candidate_pool)

        rrf_scores: dict[str, float] = {}
        for rank, chunk_id in enumerate(bm25_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)
        for rank, chunk_id in enumerate(dense_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank + 1)

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        id_to_chunk = {c.id: c for c in self.chunks}
        return [(id_to_chunk[cid], score) for cid, score in ranked_ids]


def build_default_hybrid_retriever(vector_bucket: str) -> HybridRetriever:
    corpus_dir = Path(__file__).parent.parent.parent / "corpus"
    chunks = load_corpus(corpus_dir)
    store = VectorStore(vector_bucket=vector_bucket)
    store.build(chunks)
    return HybridRetriever(chunks, store)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hybrid_retrieval.py <vector-bucket-name>")
        sys.exit(1)
    retriever = build_default_hybrid_retriever(vector_bucket=sys.argv[1])
    for q in ["What's your A/B testing experience?", "Tell me about WPP Media specifically"]:
        print(f"Query: {q!r}")
        for chunk, score in retriever.search(q, k=3):
            print(f"  [rrf={score:.4f}] {chunk.id} — {chunk.section}")
        print()