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

import hashlib
import json
import threading
import time
from dataclasses import dataclass

from vector_store import VectorStore


@dataclass
class CacheEntry:
    question: str
    answer: str
    timestamp: float


class CacheStats:
    """In-process counters for ONE warm Lambda container (they reset on a
    cold start and are not shared across containers). That is enough for
    the /health snapshot; the fleet-wide numbers come from the structured
    `cache_lookup` log lines and the CloudWatch metrics built from them."""

    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.stale = 0          # similar enough, but older than the TTL
        self.saved_ms = 0
        self.saved_tokens = 0

    def record(self, outcome: str, saved_ms: int = 0, saved_tokens: int = 0):
        with self._lock:
            if outcome == "hit":
                self.hits += 1
                self.saved_ms += saved_ms
                self.saved_tokens += saved_tokens
            elif outcome == "stale":
                self.stale += 1
                self.misses += 1
            else:
                self.misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "lookups": lookups,
                "hits": self.hits,
                "misses": self.misses,
                "stale": self.stale,
                "hit_rate": round(self.hits / lookups, 3) if lookups else None,
                "saved_ms": self.saved_ms,
                "saved_tokens": self.saved_tokens,
            }


@dataclass
class CacheLookup:
    answer: str | None
    outcome: str            # "hit" | "miss" | "stale" | "empty"
    similarity: float | None = None
    saved_ms: int = 0       # how long the ORIGINAL answer took to produce
    saved_tokens: int = 0   # how many LLM tokens the original answer used


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
        self.stats = CacheStats()
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

    def lookup(self, question: str) -> CacheLookup:
        """Like get(), but says WHY it hit or missed and what a hit saved.
        Every call also updates the in-process stats and prints one
        structured `cache_lookup` JSON line (CloudWatch Logs Insights can
        query it; no new AWS permissions needed)."""
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
            return self._finish(CacheLookup(None, "empty"))

        top = results[0]
        # cosine distance -> similarity (S3 Vectors returns distance, not similarity directly)
        similarity = round(1 - top["distance"], 4)
        if similarity < self.similarity_threshold:
            return self._finish(CacheLookup(None, "miss", similarity))

        metadata = top["metadata"]
        if time.time() - metadata["timestamp"] > self.ttl_seconds:
            # stale entry -- treat as a miss rather than returning an outdated answer
            return self._finish(CacheLookup(None, "stale", similarity))

        return self._finish(CacheLookup(
            metadata["answer"], "hit", similarity,
            saved_ms=int(metadata.get("gen_ms", 0)),
            saved_tokens=int(metadata.get("gen_tokens", 0)),
        ))

    def _finish(self, result: CacheLookup) -> CacheLookup:
        self.stats.record(result.outcome, result.saved_ms, result.saved_tokens)
        print(json.dumps({
            "event": "cache_lookup",
            "outcome": result.outcome,
            "similarity": result.similarity,
            "saved_ms": result.saved_ms,
            "saved_tokens": result.saved_tokens,
        }))
        return result

    def get(self, question: str) -> str | None:
        return self.lookup(question).answer

    def put(self, question: str, answer: str, gen_ms: int = 0, gen_tokens: int = 0):
        """gen_ms / gen_tokens record what producing this answer cost, so a
        later hit can report how much time and how many tokens it saved."""
        embedding = self.vector_store.embed([question])[0]
        # Key is a hash of the question text -- deterministic, so re-caching
        # the same question overwrites its own prior entry rather than
        # accumulating duplicates. (sha256, NOT the builtin hash(): Python
        # randomizes hash() per process, so every Lambda container would
        # have produced a different key for the same question.)
        key = "cache-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:32]
        self.vector_store._s3vectors.put_vectors(
            vectorBucketName=self.vector_bucket,
            indexName=self.index_name,
            vectors=[{
                "key": key,
                "data": {"float32": embedding.tolist()},
                "metadata": {
                    "question": question, "answer": answer, "timestamp": time.time(),
                    "gen_ms": int(gen_ms), "gen_tokens": int(gen_tokens),
                },
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