"""CI evaluation gate for the twin.

Runs a fixed golden question set through the REAL pipeline (real index, real
models) and fails (exit 1) if quality regresses. Three signals:

  pass_rate     deterministic checks per question: required facts present
                (must_contain_any), forbidden text absent (must_not_contain)
  faithfulness  RAGAS-style: a Haiku judge checks whether every factual claim
                in the answer is supported by the corpus. Reported as the
                fraction of answers judged fully supported.
  p95_latency_s 95th-percentile wall time per question

Thresholds live in eval/golden.json so changing the bar is a reviewed diff.
The semantic cache is bypassed -- otherwise a cached answer from an earlier
deploy would hide a regression in the code under test.

Usage (from repo root, with AWS creds + ANTHROPIC_API_KEY set):
  python3 eval/run_eval.py <vector-bucket-name>
"""
import glob
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "backend"))

JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROMPT = """You are grading an AI assistant's answer for faithfulness.

SOURCE DOCUMENTS (the only ground truth):
{corpus}

QUESTION: {question}
ANSWER: {answer}

Is every factual claim in the ANSWER supported by the SOURCE DOCUMENTS? Statements about what the
assistant cannot help with, or polite redirects, count as supported. Reply with exactly one word:
SUPPORTED or UNSUPPORTED."""


def check_case(case: dict, answer: str) -> list[str]:
    """Returns a list of failure reasons (empty list = pass)."""
    failures = []
    lowered = answer.lower()
    must = case.get("must_contain_any")
    if must and not any(m.lower() in lowered for m in must):
        failures.append(f"missing all of {must}")
    for banned in case.get("must_not_contain", []):
        if banned.lower() in lowered:
            failures.append(f"contains forbidden text {banned!r}")
    return failures


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def summarize(results: list[dict], thresholds: dict) -> dict:
    total = len(results)
    passed = sum(1 for r in results if not r["failures"])
    judged = [r for r in results if r["faithful"] is not None]
    faithful = sum(1 for r in judged if r["faithful"])
    summary = {
        "pass_rate": passed / total if total else 0.0,
        "faithfulness": faithful / len(judged) if judged else 1.0,
        "p95_latency_s": p95([r["latency_s"] for r in results]),
    }
    summary["gate_failures"] = []
    if summary["pass_rate"] < thresholds["pass_rate"]:
        summary["gate_failures"].append(f"pass_rate {summary['pass_rate']:.2f} < {thresholds['pass_rate']}")
    if summary["faithfulness"] < thresholds["faithfulness"]:
        summary["gate_failures"].append(f"faithfulness {summary['faithfulness']:.2f} < {thresholds['faithfulness']}")
    if summary["p95_latency_s"] > thresholds["p95_latency_s"]:
        summary["gate_failures"].append(f"p95 latency {summary['p95_latency_s']:.1f}s > {thresholds['p95_latency_s']}s")
    return summary


def load_corpus() -> str:
    parts = []
    for path in sorted(glob.glob(os.path.join(ROOT, "corpus", "*.md"))):
        with open(path, encoding="utf-8") as f:
            parts.append(f"## {os.path.basename(path)}\n{f.read()}")
    return "\n\n".join(parts)


def judge_faithful(client, corpus: str, question: str, answer: str) -> bool:
    response = client.messages.create(
        model=JUDGE_MODEL, max_tokens=5,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(corpus=corpus, question=question, answer=answer)}],
    )
    return response.content[0].text.strip().upper().startswith("SUPPORTED")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    from backend.pipeline import TwinRAGPipeline

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.json")) as f:
        golden = json.load(f)

    pipeline = TwinRAGPipeline(vector_bucket=sys.argv[1])
    pipeline.connect()
    pipeline.cache.get = lambda q: None      # bypass cache both ways
    pipeline.cache.put = lambda q, a: None

    corpus = load_corpus()
    results = []
    for case in golden["cases"]:
        start = time.monotonic()
        try:
            answer = pipeline.answer(case["question"]).answer
            error = None
        except Exception as e:  # a crash is a failed case, not a crashed gate
            answer, error = "", f"{type(e).__name__}: {e}"
        latency = time.monotonic() - start

        failures = [error] if error else check_case(case, answer)
        faithful = None
        if not error and not case.get("skip_faithfulness"):
            faithful = judge_faithful(pipeline.client, corpus, case["question"], answer)
        results.append({"id": case["id"], "answer": answer, "failures": failures,
                        "faithful": faithful, "latency_s": round(latency, 2)})
        status = "PASS" if not failures and faithful is not False else "FAIL"
        print(f"[{status}] {case['id']:<22} {latency:5.1f}s  faithful={faithful}  {failures or ''}")

    summary = summarize(results, golden["thresholds"])
    report = {"summary": summary, "results": results}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}, indent=2))
    if summary["gate_failures"]:
        print("EVAL GATE FAILED:", "; ".join(summary["gate_failures"]))
        sys.exit(1)
    print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()
