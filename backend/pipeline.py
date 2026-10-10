"""
The full RAG pipeline, wiring together every piece into one end-to-end
flow. This is what the Lambda handler calls per request.

Flow per query:
  1. Check semantic cache -> if hit, return immediately, skip everything else.
  2. Generate a HyDE hypothetical document from the question.
  3. Hybrid-retrieve (BM25 + dense + RRF) using the HyDE doc as the search
     query -- retrieval runs over BOTH original leaf chunks AND RAPTOR
     summary nodes, so both specific-fact and synthesis-style questions
     are covered by the same S3 Vectors index.
  4. Corrective grading: drop any retrieved passage the LLM says isn't
     actually relevant.
  5. If nothing survives grading, abstain honestly rather than force an
     answer from bad context.
  6. Otherwise, generate the final answer from the surviving context.
  7. Cache the (question, answer) pair for next time.

Important architectural note, different from an earlier version: this file
used to have a build() method that computed embeddings and built a RAPTOR
tree LIVE, every time a Lambda container cold-started. That's gone now --
all of that (contextual retrieval, RAPTOR, embedding, uploading to S3
Vectors) happens OFFLINE in build_index.py, run manually whenever the
corpus changes. This file's connect() is deliberately lightweight: it just
wires up a retriever and cache against whatever's ALREADY in S3 Vectors,
with zero embedding or LLM calls of its own. Cold starts are fast because
of this separation, not because of any caching trick within Lambda itself.
"""

import os
import sys
from dataclasses import dataclass

from anthropic import Anthropic

from chassis.llm import build_client
from chassis.prompts import get_prompt
from chassis.tracing import record_usage, traced_llm_call

# rag/ modules use simple (non-package) imports internally (e.g. `from chunking
# import Chunk`), so we add that directory to the path rather than importing
# via a package prefix — keeps every rag/ file runnable standalone AND
# importable from here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rag"))

from rag.corrective_rag import CorrectiveGrader
from rag.hybrid_retrieval import HybridRetriever
from rag.hyde import HyDEGenerator
from rag.semantic_cache import SemanticCache
from rag.vector_store import VectorStore


# Conversation history is capped before it reaches ANY prompt. Measured from
# real Langfuse traces: condense was sending ~4,300 input tokens (mostly
# earlier assistant replies) to produce a ~13-token rewrite, and took 12-15s;
# calls with ~130-270 input tokens (hyde, grading) took 0.5-1.3s. Combined
# with final generation, full requests ran 32-41s, over API Gateway's ~29s
# limit. Two recent turns is enough to resolve "that"/"it"/"those".
MAX_HISTORY_MESSAGES = 10   # last 2 user/assistant turns
MAX_HISTORY_CHARS = 500    # per message


def _trim_history(history: list[dict]) -> list[dict]:
    trimmed = []
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        content = turn["content"]
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS].rstrip() + "..."
        trimmed.append({"role": turn["role"], "content": content})
    return trimmed


@dataclass
class PipelineResult:
    answer: str
    from_cache: bool
    sources: list[str]  # chunk/node ids actually used, for debugging/transparency


@dataclass
class _Plan:
    """What _prepare() decided: a cache hit, or the material for generation
    (context is None when nothing relevant was retrieved)."""
    standalone_question: str
    cached: str | None = None
    history_block: str = ""
    context: str | None = None
    sources: list[str] | None = None


class TwinRAGPipeline:
    def __init__(self, vector_bucket: str, client: Anthropic | None = None, model: str = "claude-sonnet-4-6",
                 fast_model: str = "claude-haiku-4-5-20251001"):
        """Two models, deliberately -- `model` (Sonnet-class) is used ONLY
        where the user directly reads the output: the final answer and the
        no-context/greeting response. `fast_model` (Haiku-class) handles
        the internal-only steps whose output is never shown to anyone --
        query condensation, HyDE's hypothetical document, and corrective
        grading's yes/no verdicts. This was a real, measured fix: a single
        request chains together up to 4 sequential LLM calls (condense,
        HyDE, grading, final generation), and every one of them was
        previously running on the same, slower model regardless of how
        trivial the actual task was. Swapping the 2-3 internal calls to a
        materially faster model cuts real wall-clock latency without
        touching the quality of what a recruiter actually reads."""
        self.vector_bucket = vector_bucket
        self.client = client or build_client(os.environ["ANTHROPIC_API_KEY"])
        self.model = model
        self.fast_model = fast_model

        self.hyde = HyDEGenerator(client=self.client, model=fast_model)
        self.grader = CorrectiveGrader(client=self.client, model=fast_model)

        self.vector_store: VectorStore | None = None
        self.retriever: HybridRetriever | None = None
        self.cache: SemanticCache | None = None
        self._connected = False

    def connect(self):
        """The only setup step Lambda ever runs -- connects to the S3
        Vectors index build_index.py already populated. No embedding, no
        RAPTOR, no corpus loading: list_all_chunks() just reads text back
        out of vectors that already exist, needed for BM25's in-memory
        keyword index (see vector_store.py's own docstring for why BM25
        specifically needs this, unlike dense search)."""
        self.vector_store = VectorStore(vector_bucket=self.vector_bucket, index_name="corpus")
        all_chunks = self.vector_store.list_all_chunks()
        self.retriever = HybridRetriever(all_chunks, self.vector_store)
        self.cache = SemanticCache(self.vector_store)
        self._connected = True

    @traced_llm_call("condense")
    def _condense_question(self, question: str, history: list[dict]) -> str:
        """Rewrites a follow-up question into a standalone one, using
        conversation history to resolve references like "that" or "it".
        Only called when history is non-empty -- with no history, the
        question is already standalone by definition."""
        history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        prompt = get_prompt("condense")
        response = self.client.messages.create(
            model=self.fast_model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt.format(history=history_text, question=question)}],
        )
        record_usage(response, prompt)
        return response.content[0].text.strip()

    def _prepare(self, question: str, history: list[dict] | None, k: int) -> "_Plan":
        """Everything BEFORE the final text is generated: condense, cache
        lookup, HyDE, retrieval, grading. Shared by answer() and
        answer_stream() so the two cannot drift apart."""
        if not self._connected:
            raise RuntimeError("Call connect() before answer() -- pipeline has no index connection yet.")

        history = _trim_history(history or [])

        # Condense to a standalone question BEFORE caching, HyDE, or
        # retrieval -- all three need to know what "that" actually refers
        # to, not just the final answer-generation step. This also keeps
        # the semantic cache correct: caching keys on the condensed,
        # standalone question, not the literal follow-up phrasing, so two
        # different follow-ups that happen to share wording but different
        # actual context don't collide in the cache.
        standalone_question = self._condense_question(question, history) if history else question

        cached = self.cache.get(standalone_question)
        if cached is not None:
            return _Plan(standalone_question=standalone_question, cached=cached)

        hypothetical_doc = self.hyde.generate(standalone_question)
        retrieved = self.retriever.search(hypothetical_doc, k=k)
        candidate_chunks = [chunk for chunk, _ in retrieved]

        relevant_chunks = self.grader.filter_relevant(standalone_question, candidate_chunks)

        history_block = ""
        if history:
            history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
            history_block = f"Conversation so far:\n{history_text}\n"

        context = "\n\n---\n\n".join(c.text for c in relevant_chunks) if relevant_chunks else None
        return _Plan(
            standalone_question=standalone_question,
            history_block=history_block,
            context=context,
            sources=[c.id for c in relevant_chunks],
        )

    def answer(self, question: str, history: list[dict] | None = None, k: int = 4) -> PipelineResult:
        plan = self._prepare(question, history, k)
        if plan.cached is not None:
            return PipelineResult(answer=plan.cached, from_cache=True, sources=[])

        if plan.context is None:
            # No grounded content matched -- but that doesn't mean a rigid
            # canned string is the right response. A real LLM call here
            # (still constrained to NOT invent specific facts) can tell
            # "this is just a greeting" apart from "this is a real question
            # I genuinely don't have information for" -- the old fixed
            # ABSTAIN_MESSAGE answered both cases identically, which read
            # as broken on something as simple as "hello".
            answer = self._generate_no_context_response(plan.history_block, question)
        else:
            # The final answer uses the REAL history and the ORIGINAL
            # question (not the condensed one) -- this keeps the reply
            # feeling like a natural continuation of the actual
            # conversation, rather than an answer to a question the
            # recruiter never literally typed.
            answer = self._generate_final_answer(plan.history_block, plan.context, question)

        self.cache.put(plan.standalone_question, answer)
        return PipelineResult(answer=answer, from_cache=False, sources=plan.sources)

    def answer_stream(self, question: str, on_delta, history: list[dict] | None = None,
                      k: int = 4) -> PipelineResult:
        """Same flow as answer(), but the final text is delivered piece by
        piece through on_delta(text) as the model writes it. Blocking: returns
        the complete PipelineResult at the end. Everything before the final
        generation (condense/HyDE/grading) is unchanged and not streamed."""
        plan = self._prepare(question, history, k)
        if plan.cached is not None:
            on_delta(plan.cached)
            return PipelineResult(answer=plan.cached, from_cache=True, sources=[])

        if plan.context is None:
            answer = self._generate_no_context_response_stream(plan.history_block, question, on_delta)
        else:
            answer = self._generate_final_answer_stream(plan.history_block, plan.context, question, on_delta)

        self.cache.put(plan.standalone_question, answer)
        return PipelineResult(answer=answer, from_cache=False, sources=plan.sources)

    def _complete(self, on_delta, **kwargs):
        """Runs one generation, streaming when the client supports it and
        falling back to a single create() (delivered as one chunk) when not --
        e.g. the plain Anthropic client or test fakes."""
        if on_delta is not None and hasattr(self.client, "stream"):
            return self.client.stream(on_delta, **kwargs)
        response = self.client.messages.create(**kwargs)
        if on_delta is not None:
            on_delta(response.content[0].text)
        return response

    @traced_llm_call("no_context_response")
    def _generate_no_context_response(self, history_block: str, question: str) -> str:
        prompt = get_prompt("no_context")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt.format(
                history_block=history_block, question=question,
            )}],
        )
        record_usage(response, prompt)
        return response.content[0].text.strip()

    @traced_llm_call("no_context_response")
    def _generate_no_context_response_stream(self, history_block: str, question: str, on_delta) -> str:
        prompt = get_prompt("no_context")
        response = self._complete(
            on_delta,
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt.format(
                history_block=history_block, question=question,
            )}],
        )
        record_usage(response, prompt)
        return response.content[0].text.strip()

    @traced_llm_call("final_generation")
    def _generate_final_answer(self, history_block: str, context: str, question: str) -> str:
        prompt = get_prompt("answer")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt.format(
                history_block=history_block, context=context, question=question,
            )}],
        )
        record_usage(response, prompt)
        return response.content[0].text.strip()

    @traced_llm_call("final_generation")
    def _generate_final_answer_stream(self, history_block: str, context: str, question: str, on_delta) -> str:
        prompt = get_prompt("answer")
        response = self._complete(
            on_delta,
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt.format(
                history_block=history_block, context=context, question=question,
            )}],
        )
        record_usage(response, prompt)
        return response.content[0].text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <vector-bucket-name>")
        print("(Run build_index.py against this bucket first, or this will find an empty index.)")
        sys.exit(1)

    pipeline = TwinRAGPipeline(vector_bucket=sys.argv[1])
    print("Connecting to S3 Vectors index (fast -- no embedding, no RAPTOR)...")
    pipeline.connect()
    print("Connected.\n")

    test_questions = [
        "What's your experience with A/B testing?",
        "Can you summarize your overall career?",
        "What's your favorite food?",
        "What's your experience with A/B testing?",  # repeat -> should hit semantic cache
    ]
    for q in test_questions:
        result = pipeline.answer(q)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        print(f"(from_cache={result.from_cache}, sources={result.sources})\n")