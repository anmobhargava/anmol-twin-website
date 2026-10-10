"""
Corrective / Self-RAG grading.

The problem it solves: retrieval isn't perfect — even hybrid BM25+dense+RRF
can return a chunk that's topically adjacent but doesn't actually answer the
question. Feeding an irrelevant chunk straight to the generator risks a
confidently-wrong or off-topic answer.

The fix: after retrieval, ask the LLM itself to grade each retrieved chunk —
"does this actually help answer the question, yes/no?" — and DROP chunks that
fail the grade before generation ever sees them. This is a real, if small,
extra LLM call per query; the trade is slightly higher latency/cost for
meaningfully higher answer quality on edge-case or ambiguous questions.

If ALL chunks get graded as irrelevant, we fall back to an explicit
"I don't have specific information about that" rather than forcing a
generation from bad context — this is the "self-RAG" abstention behavior.
"""

import contextvars
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic

# chassis/ lives at backend/chassis/ (a vendored copy of tracing.py, one
# level up from this file) -- see hyde.py's own comment for why this path
# insert is needed for both standalone and pipeline-imported execution.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chassis.prompts import get_prompt
from chassis.tracing import record_usage, traced_llm_call
from chunking import Chunk


class CorrectiveGrader:
    def __init__(self, client: Anthropic | None = None, model: str = "claude-sonnet-4-6"):
        self.client = client or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    @traced_llm_call("grading")
    def grade(self, question: str, chunk: Chunk) -> bool:
        prompt = get_prompt("grade")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt.format(question=question, passage=chunk.text)}],
        )
        record_usage(response, prompt)
        verdict = response.content[0].text.strip().upper()
        return verdict.startswith("YES")

    def filter_relevant(self, question: str, chunks: list[Chunk]) -> list[Chunk]:
        """Grades all candidate chunks CONCURRENTLY, not one at a time.
        Each chunk's grade is independent of every other chunk's -- grading
        chunk A never needs chunk B's result -- so there's no reason to wait
        for one call to finish before starting the next. This was a real,
        measured latency problem: with up to 4 retrieved chunks, sequential
        grading meant up to 4 full round-trips to Claude stacked back to
        back, on top of condensation + HyDE + final generation. Parallel
        grading turns that into roughly the time of the SLOWEST single
        grading call, not the sum of all of them.

        Results are reassembled in the ORIGINAL retrieval-rank order
        (not completion order, which is nondeterministic) -- parallelizing
        the calls and preserving output order are separate concerns, and
        keeping the original rank order matters for what the final
        generation prompt sees first."""
        if not chunks:
            return []
        # contextvars.copy_context() matters here specifically because of
        # @traced_llm_call on grade(): Langfuse's trace context (which
        # observation is "current") is carried via Python contextvars, and
        # contextvars do NOT automatically propagate into a NEW OS thread
        # the way they do into an asyncio task -- a bare `executor.map`
        # would run each grade() call in a thread with a fresh, EMPTY
        # context, so each parallel grading generation would show up in
        # Langfuse as its own orphaned root trace instead of nesting under
        # this request's actual agent_run/traced_llm_call span tree.
        #
        # IMPORTANT, found via real production traces (43% of requests
        # with >=2 graded chunks were hitting
        # "RuntimeError: cannot enter context: <Context> is already
        # entered"): a single contextvars.Context object can only be
        # entered in ONE place at a time -- across threads too, not just
        # recursively. The original code called `ctx.run(...)` with the
        # SAME captured `ctx` from every worker thread simultaneously,
        # which is exactly the forbidden concurrent-entry case: whenever
        # two grade() calls' API round-trips overlapped in time (near
        # certain with >=2 chunks dispatched together), the second thread
        # to call ctx.run() raised. That exception then propagated all the
        # way out of filter_relevant() -> pipeline.answer() -> the
        # lambda_handler's `with agent_run(...)` block, surfacing to the
        # recruiter as an actual 500 response, not just a tracing gap.
        #
        # Fix: give each parallel task its OWN Context object that carries
        # the same captured values (ctx.copy()), instead of making every
        # thread fight over entering the identical one. Copies are cheap
        # (shallow) and each can be entered independently and concurrently.
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            verdicts = list(executor.map(lambda c: ctx.copy().run(self.grade, question, c), chunks))
        return [c for c, is_relevant in zip(chunks, verdicts) if is_relevant]


if __name__ == "__main__":
    import sys
    from hybrid_retrieval import build_default_hybrid_retriever

    if len(sys.argv) < 2:
        print("Usage: python corrective_rag.py <vector-bucket-name>")
        sys.exit(1)

    retriever = build_default_hybrid_retriever(vector_bucket=sys.argv[1])
    grader = CorrectiveGrader()

    question = "Do you have experience with cooking?"  # deliberately off-corpus-focus, to test grading
    results = retriever.search(question, k=3)
    print(f"Question: {question!r}")
    print("Retrieved (pre-grading):")
    for chunk, score in results:
        print(f"  {chunk.id} — {chunk.section}")

    graded = grader.filter_relevant(question, [c for c, _ in results])
    print(f"\nAfter grading, {len(graded)}/{len(results)} chunks kept as relevant:")
    for c in graded:
        print(f"  {c.id} — {c.section}")