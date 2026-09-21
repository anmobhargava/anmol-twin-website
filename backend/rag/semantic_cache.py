"""
Semantic caching, backed by S3 Vectors.

The problem it solves: recruiters browsing a twin chatbot will ask
near-duplicate questions constantly — "what's your AI experience?",
"tell me about your AI work", "do you know AI?" are functionally the same
question worded differently. A naive exact-string cache would miss all of
these; re-running the full retrieval + generation pipeline for each wastes
both latency and Anthropic API cost for genuinely repeated questions.

Why S3 Vectors instead of an earlier in-memory version: a cached question
is just another vector, and "find a previously-asked question similar to
this one" is the exact same similarity-search operation S3 Vectors already
does for retrieval -- reusing it here means the cache is now genuinely
persistent (survives cold starts) and shared across every concurrent
Lambda instance, not scoped to one warm container's memory. Stored in a
separate S3 Vectors INDEX from the corpus/RAPTOR data (same vector bucket
as the passed-in VectorStore, different index) so cache entries and
retrieval content never mix in a search.
"""

import time
from dataclasses import dataclass

from vector_store import VectorStore


@dataclass
class CacheEntry:
    question: str
    answer: str
    timestamp: float


class SemanticCache:
    def __init__(self, vector_store: VectorStore, index_name: str = "qa-cache",
                 similarity_threshold: float = 0.92, ttl_seconds: int = 3600):
        """Takes the vector_store's OWN bucket (vector_store.vector_bucket)
        rather than a separately-passed bucket name -- passing a different
        bucket than the one vector_store actually uses would silently
        create an inconsistency (cache entries in one bucket, corpus data
        in another) with no error to catch it.

        similarity_threshold=0.92 is intentionally strict — we want near-
        DUPLICATE questions to hit the cache, not just topically-related ones,
        since a wrong cache hit means giving a stale or mismatched answer.
        ttl_seconds=3600 (1hr) bounds how long a cached answer stays valid --
        S3 Vectors has no native per-item TTL, so this is enforced manually
        by storing a timestamp in each entry's metadata and checking it at
        read time."""
        self.vector_store = vector_store
        self.vector_bucket = vector_store.vector_bucket
        self.index_name = index_name
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Creates the cache's own S3 Vectors index if it doesn't exist yet
        -- separate from the corpus/RAPTOR retrieval index, since mixing
        cached Q&A pairs into the same index as actual corpus content would
        let a cache entry accidentally get returned as a "retrieved chunk"."""
        try:
            self.vector_store._s3vectors.create_index(
                vectorBucketName=self.vector_bucket,
                indexName=self.index_name,
                dataType="float32",
                dimension=self.vector_store.dimension,
                distanceMetric="cosine",
            )
        except self.vector_store._s3vectors.exceptions.ConflictException:
            pass  # index already exists, nothing to do

    def get(self, question: str) -> str | None:
        query_embedding = self.vector_store.embed([question])[0]
        response = self.vector_store._s3vectors.query_vectors(
            vectorBucketName=self.vector_bucket,
            indexName=self.index_name,
            queryVector={"float32": query_embedding.tolist()},
            topK=1,
            returnDistance=True,
            returnMetadata=True,
        )
        results = response.get("vectors", [])
        if not results:
            return None

        top = results[0]
        # cosine distance -> similarity (S3 Vectors returns distance, not similarity directly)
        similarity = 1 - top["distance"]
        if similarity < self.similarity_threshold:
            return None

        metadata = top["metadata"]
        if time.time() - metadata["timestamp"] > self.ttl_seconds:
            return None  # stale entry -- treat as a miss rather than returning an outdated answer

        return metadata["answer"]

    def put(self, question: str, answer: str):
        embedding = self.vector_store.embed([question])[0]
        # Key is a hash of the question text -- deterministic, so re-caching
        # the same question overwrites its own prior entry rather than
        # accumulating duplicates.
        key = f"cache-{abs(hash(question))}"
        self.vector_store._s3vectors.put_vectors(
            vectorBucketName=self.vector_bucket,
            indexName=self.index_name,
            vectors=[{
                "key": key,
                "data": {"float32": embedding.tolist()},
                "metadata": {"question": question, "answer": answer, "timestamp": time.time()},
            }],
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python semantic_cache.py <vector-bucket-name>")
        sys.exit(1)

    store = VectorStore(vector_bucket=sys.argv[1])
    cache = SemanticCache(store)
    cache.put("What's your AI experience?", "I've built RAG pipelines, fine-tuned models, and production infra.")

    test_questions = [
        "What's your AI experience?",       # exact match -> should hit
        "Tell me about your AI work",        # near-duplicate -> should hit
        "What's your favorite food?",        # unrelated -> should miss
    ]
    for q in test_questions:
        result = cache.get(q)
        status = "HIT" if result else "MISS"
        print(f"[{status}] {q!r}")