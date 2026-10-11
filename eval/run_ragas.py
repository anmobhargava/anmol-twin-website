"""RAGAS evaluation of the twin's RAG pipeline (report-only, does NOT gate deploys).

eval/run_eval.py is the fast, cheap deploy gate. This is the deeper, slower
measurement that tracks retrieval + generation quality over time:

  faithfulness        is every claim in the answer supported by the retrieved
                      context? (hallucination check, scored per claim)
  context_precision   how much of the retrieved context was actually useful
                      for the question? (retrieval noise; no reference needed)
  context_recall      did retrieval cover everything the reference answer
                      needs? Only computed for golden cases that have a
                      "reference" field.

The judge LLM is Claude Haiku. Cases marked skip_faithfulness (off-topic /
abstention checks) are excluded: they are designed to have no grounding.
The semantic cache is bypassed so scores reflect the current code.

Usage (repo root, AWS creds + ANTHROPIC_API_KEY set; needs eval/requirements-ragas.txt):
  python3 eval/run_ragas.py <vector-bucket-name>
Writes eval/ragas_report.json.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))

JUDGE_MODEL = "claude-haiku-4-5-20251001"


def build_samples(cases: list[dict], results: dict[str, dict]) -> list[dict]:
    """Turns golden cases + pipeline outputs into RAGAS sample dicts. Cases
    that errored, were skipped, or retrieved no context have nothing to
    grade and are left out (they are counted separately in the report)."""
    samples = []
    for case in cases:
        res = results.get(case["id"])
        if case.get("skip_faithfulness") or not res or res.get("error") or not res.get("contexts"):
            continue
        sample = {
            "id": case["id"],
            "user_input": case["question"],
            "response": res["answer"],
            "retrieved_contexts": res["contexts"],
        }
        if case.get("reference"):
            sample["reference"] = case["reference"]
        samples.append(sample)
    return samples


def mean(values: list) -> float | None:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return round(sum(clean) / len(clean), 3) if clean else None


def aggregate(rows: list[dict]) -> dict:
    metrics = sorted({k for r in rows for k in r if k != "id"})
    return {m: mean([r.get(m) for r in rows]) for m in metrics}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    from backend.pipeline import TwinRAGPipeline
    from backend.rag.semantic_cache import CacheLookup

    with open(os.path.join(HERE, "golden.json")) as f:
        golden = json.load(f)

    pipeline = TwinRAGPipeline(vector_bucket=sys.argv[1])
    pipeline.connect()
    pipeline.cache.lookup = lambda q: CacheLookup(None, "miss")
    pipeline.cache.get = lambda q: None
    pipeline.cache.put = lambda q, a, **kw: None

    results = {}
    for case in golden["cases"]:
        start = time.monotonic()
        try:
            r = pipeline.answer(case["question"])
            results[case["id"]] = {"answer": r.answer, "contexts": r.contexts or [], "error": None}
        except Exception as e:
            results[case["id"]] = {"answer": "", "contexts": [], "error": f"{type(e).__name__}: {e}"}
        print(f"ran {case['id']:<22} {time.monotonic() - start:5.1f}s")

    samples = build_samples(golden["cases"], results)
    print(f"\n{len(samples)} of {len(golden['cases'])} cases are gradable by RAGAS")
    if not samples:
        sys.exit("nothing to grade")

    # Imported here so unit tests of build_samples/aggregate don't need ragas.
    from langchain_anthropic import ChatAnthropic
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall

    judge = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0, max_tokens=2048))
    metrics = [Faithfulness(llm=judge), LLMContextPrecisionWithoutReference(llm=judge)]
    with_reference = [s for s in samples if "reference" in s]
    ids = [s.pop("id") for s in samples]

    scored = evaluate(EvaluationDataset.from_list(samples), metrics=metrics, llm=judge).to_pandas()
    rows = [{"id": i, **{k: float(v) for k, v in row.items() if k in ("faithfulness", "llm_context_precision_without_reference")}}
            for i, (_, row) in zip(ids, scored.iterrows())]

    if with_reference:
        ref_samples = [{k: v for k, v in s.items()} for s in with_reference]
        rec = evaluate(EvaluationDataset.from_list(ref_samples), metrics=[LLMContextRecall(llm=judge)], llm=judge).to_pandas()
        recall_by_q = dict(zip(rec["user_input"], rec["context_recall"]))
        for row, s in zip(rows, samples):
            if s["user_input"] in recall_by_q:
                row["context_recall"] = float(recall_by_q[s["user_input"]])

    report = {"judge": JUDGE_MODEL, "n_graded": len(rows), "n_cases": len(golden["cases"]),
              "aggregate": aggregate(rows), "per_case": rows}
    with open(os.path.join(HERE, "ragas_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["aggregate"], indent=2))

    sys.path.insert(0, HERE)
    import tracking
    tracking.log_run(
        "ragas",
        tracking.default_params(job="ragas", judge=JUDGE_MODEL, n_graded=len(rows)),
        {"ragas": report["aggregate"]},
        [os.path.join(HERE, "ragas_report.json")],
    )


if __name__ == "__main__":
    main()
