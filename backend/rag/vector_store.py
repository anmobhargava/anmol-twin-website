"""
S3 Vectors-backed vector store for the twin's RAG corpus.

This replaces an earlier FAISS-based version. Two real reasons for the
switch: (1) FAISS meant managing a serialized index file ourselves (save
to S3, download to /tmp, load into memory on every cold start) -- S3
Vectors removes all of that, since embeddings just live there permanently
and get queried directly, no local index to rebuild or reload; (2) it
sets up a clean separation between BUILDING the index (build_index.py,
run manually/by CI whenever the corpus changes) and SERVING requests
(Lambda only ever queries, never re-embeds or rebuilds anything).

One real tradeoff, stated honestly: hybrid retrieval's BM25 half needs
actual chunk TEXT in memory (not just embeddings) for keyword search --
S3 Vectors doesn't replace that. list_all_chunks() below fetches text back
out of stored metadata at cold start for exactly this reason. This is a
cheap S3 read, not a Bedrock/Claude call, so it's still a real improvement
over the old FAISS approach, just not a 100% free cold start.
"""

import json
import os

import boto3
import numpy as np

from chunking import Chunk

EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024  # Titan V2 supports 256, 512, or 1024 -- using AWS's own documented default
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")


class VectorStore:
    def __init__(self, vector_bucket: str, index_name: str = "corpus",
                 model_id: str = EMBEDDING_MODEL_ID, dimension: int = EMBEDDING_DIM):
        self.model_id = model_id
        self.dimension = dimension
        self.vector_bucket = vector_bucket
        self.index_name = index_name
        self._bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        self._s3vectors = boto3.client("s3vectors", region_name=BEDROCK_REGION)
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Idempotent -- safe to call every time a VectorStore is
        constructed (including on every Lambda cold start), not just
        during build_index.py's one-time setup."""
        try:
            self._s3vectors.create_index(
                vectorBucketName=self.vector_bucket,
                indexName=self.index_name,
                dataType="float32",
                dimension=self.dimension,
                distanceMetric="cosine",
            )
        except self._s3vectors.exceptions.ConflictException:
            pass  # index already exists, nothing to do

    def _embed_one(self, text: str) -> list[float]:
        """A single Bedrock InvokeModel call. Titan's API embeds one text
        per request -- there is no batch-embedding endpoint via InvokeModel."""
        body = json.dumps({
            "inputText": text,
            "dimensions": self.dimension,
            "normalize": True,
        })
        response = self._bedrock.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = [self._embed_one(t) for t in texts]
        return np.array(vectors, dtype="float32")

    def build(self, chunks: list[Chunk]):
        """Embeds and uploads all chunks to S3 Vectors. Run by
        build_index.py, NOT at Lambda cold start -- this is what populates
        the persistent index once (or whenever the corpus changes), so
        serving never has to re-embed anything."""
        vectors_payload = []
        for chunk in chunks:
            embedding = self._embed_one(chunk.text)
            # S3 Vectors metadata values must be simple types (str/int/float/bool)
            # -- non-simple values in chunk.metadata (e.g. RAPTOR's "children"
            # list) are dropped here rather than sent as-is, since the API
            # would reject them.
            simple_metadata = {k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))}
            vectors_payload.append({
                "key": chunk.id,
                "data": {"float32": embedding},
                "metadata": {
                    "source": chunk.source,
                    "section": chunk.section,
                    "text": chunk.text,
                    **simple_metadata,
                },
            })

        # put_vectors accepts multiple vectors per call -- batch to keep
        # individual requests reasonably sized rather than one call per chunk.
        BATCH_SIZE = 50
        for i in range(0, len(vectors_payload), BATCH_SIZE):
            batch = vectors_payload[i:i + BATCH_SIZE]
            self._s3vectors.put_vectors(
                vectorBucketName=self.vector_bucket,
                indexName=self.index_name,
                vectors=batch,
            )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_embedding = self._embed_one(query)
        response = self._s3vectors.query_vectors(
            vectorBucketName=self.vector_bucket,
            indexName=self.index_name,
            queryVector={"float32": query_embedding},
            topK=k,
            returnDistance=True,
            returnMetadata=True,
        )
        results = []
        for item in response.get("vectors", []):
            meta = item["metadata"]
            chunk = Chunk(id=item["key"], source=meta["source"], section=meta["section"], text=meta["text"], metadata={})
            similarity = 1 - item["distance"]  # S3 Vectors returns cosine DISTANCE, not similarity
            results.append((chunk, similarity))
        return results

    def list_all_chunks(self) -> list[Chunk]:
        """Fetches every chunk's text back out of S3 Vectors -- needed at
        Lambda cold start for BM25 indexing (keyword search needs raw text
        in memory; dense search doesn't, since S3 Vectors handles that
        natively via query_vectors above without us ever loading vectors
        into local memory). Paginated, since a corpus could in principle
        exceed one page -- irrelevant at our ~30-node scale today, but
        correct regardless of scale."""
        chunks = []
        next_token = None
        while True:
            kwargs = {"vectorBucketName": self.vector_bucket, "indexName": self.index_name, "returnMetadata": True}
            if next_token:
                kwargs["nextToken"] = next_token
            response = self._s3vectors.list_vectors(**kwargs)
            for item in response.get("vectors", []):
                meta = item["metadata"]
                chunks.append(Chunk(id=item["key"], source=meta["source"], section=meta["section"], text=meta["text"], metadata={}))
            next_token = response.get("nextToken")
            if not next_token:
                break
        return chunks