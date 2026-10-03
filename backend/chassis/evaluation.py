"""
RAG quality evaluation, via RAGAS -- scored, then logged directly into
Langfuse (via create_score) rather than kept as a separate report. This is
what makes "is quality actually good, and did it regress?" answerable from
the same dashboard as cost/latency/tokens, not a parallel spreadsheet.

Judge model: OpenAI (ChatOpenAI + OpenAIEmbeddings), not Anthropic --
RAGAS's metrics need an LLM to judge "is this faithful to the context?"
and an embedding model for semantic-similarity metrics (answer relevancy).
Using a different provider than the twin website's own Claude calls is
deliberate, not an oversight: it avoids the judge model grading its own
homework with the exact same weights/biases as the model being judged.

DEPENDENCY WARNING -- read before touching requirements:
The latest ragas (0.4.x as of this writing) has a BROKEN install: it
unconditionally imports `langchain_community.chat_models.vertexai`, which
newer langchain-community releases no longer ship (Google's Vertex AI
integration moved to its own package, langchain-community's newer
versions don't re-export it the way ragas 0.4.x expects). Installing the
latest of everything therefore fails at import time with:
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
Verified working combination, confirmed by actually installing and
importing it in a clean environment (not assumed from memory):
    ragas==0.2.15
    langchain-core==0.2.43
    langchain-community==0.2.19
    langchain-openai==0.1.25
Pin these exact versions in pyproject.toml. Re-verify by literally
installing in a clean venv before bumping any of them later -- this
combination was arrived at empirically after several other combinations
(latest ragas, ragas 0.4.3 + latest langchain-*, various partial pins)
all failed at import time for genuinely different reasons each time.
"""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference

from chassis.tracing import get_client

JUDGE_MODEL = os.environ.get("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")


def _judge():
    """Builds the OpenAI judge LLM + embeddings, wrapped for RAGAS. Built
    fresh per call rather than as a module-level singleton -- avoids
    holding a live client across long-running/serverless processes where
    it might go stale, at the cost of a small amount of repeated setup."""
    llm = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))
    return llm, embeddings


def evaluate_response(question: str, answer: str, contexts: list[str]) -> dict[str, float]:
    """Scores ONE question/answer/context triple -- e.g. one turn of the
    twin website's RAG pipeline. Returns a plain dict of metric_name ->
    score (0-1), so callers don't need to know anything about RAGAS's own
    result object shape.

    Reference-free metrics only (no ground_truth/reference needed) --
    deliberate, because the twin website has no labeled "correct answer"
    dataset to compare against; these three answer the questions that
    matter without one:
      - faithfulness: does the answer actually follow from the retrieved
        context, or did the model add things not supported by it?
      - answer_relevancy: does the answer actually address the question
        asked, semantically (via embedding similarity)?
      - llm_context_precision_without_reference: of what was retrieved,
        how much was actually relevant/used?

    Usage (after a pipeline.answer() call, using its sources as contexts):
        scores = evaluate_response(question, result.answer, retrieved_texts)
        log_scores_to_langfuse(trace_id, scores)
    """
    llm, embeddings = _judge()
    sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
    dataset = EvaluationDataset(samples=[sample])
    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
    ]
    result = evaluate(dataset=dataset, metrics=metrics, show_progress=False)
    return {k: v for k, v in result._repr_dict.items()}


def log_scores_to_langfuse(trace_id: str, scores: dict[str, float]):
    """Attaches RAGAS scores to the SAME trace Langfuse already has for
    this request (from tracing.py's agent_run/traced_llm_call) -- this is
    the actual point of wiring RAGAS into Langfuse instead of keeping it
    as a standalone report: quality scores show up right next to the
    cost/latency/token data for the exact call that produced them,
    queryable and filterable the same way.

    trace_id comes from tracing.py's get_client().get_current_trace_id()
    -- call this WHILE still inside the agent_run(...) block that produced
    the response being scored, or capture the trace_id at that point to
    use here later if scoring happens out-of-band (e.g. an async eval job).
    """
    client = get_client()
    for name, value in scores.items():
        client.create_score(trace_id=trace_id, name=f"ragas_{name}", value=float(value), data_type="NUMERIC")
    client.flush()


def evaluate_batch_and_track(
    agent_name: str,
    run_name: str,
    samples: list[dict],
) -> dict[str, float]:
    """Runs evaluate_response() over a BATCH of samples (an eval sweep --
    e.g. 50 test questions run through the pipeline after a prompt
    change) and logs results at BOTH the levels that matter, deliberately
    not just one:

      - per-sample scores -> Langfuse, via log_scores_to_langfuse, IF the
        sample dict includes a 'trace_id' (so each score stays attached
        to the exact request that produced it -- unchanged from before)
      - the BATCH'S MEAN per metric -> MLflow, as a tracked run -- this is
        the piece evaluate_response()/log_scores_to_langfuse() alone
        couldn't answer: "did average faithfulness go UP or DOWN compared
        to the last time I ran this sweep." Langfuse scores one trace at
        a time; MLflow is what lets you compare SWEEPS against each other
        in its run-comparison UI, same as tracking.py's tracked_run is
        used for build_index.py runs elsewhere in the chassis.

    samples: list of dicts, each with 'question', 'answer', 'contexts',
    and optionally 'trace_id' (skip Langfuse logging for a sample if it
    has no trace_id -- e.g. an offline batch eval with no live request
    behind it).

    Usage:
        evaluate_batch_and_track(
            "agent-25-graphrag", "eval-sweep-2026-09-28",
            samples=[
                {"question": q, "answer": a, "contexts": ctx, "trace_id": tid}
                for q, a, ctx, tid in zip(questions, answers, contexts_list, trace_ids)
            ],
        )
    """
    from chassis.tracking import log_metrics, tracked_run

    all_scores = []
    for sample in samples:
        scores = evaluate_response(sample["question"], sample["answer"], sample["contexts"])
        all_scores.append(scores)
        if sample.get("trace_id"):
            log_scores_to_langfuse(sample["trace_id"], scores)

    metric_names = all_scores[0].keys()
    means = {
        f"ragas_{name}_mean": sum(s[name] for s in all_scores) / len(all_scores)
        for name in metric_names
    }

    with tracked_run(agent_name, run_name, params={"n_samples": len(samples)}):
        log_metrics(means)

    return means