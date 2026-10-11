"""MLflow experiment tracking for the evaluation jobs (fail-open).

Every eval run becomes one MLflow run: parameters (git sha, prompt source/
label, models), metrics (pass_rate, faithfulness, p95 latency, RAGAS scores)
and the JSON report as an artifact. That makes "did quality change between
these two deploys?" a comparison in the MLflow UI instead of digging through
CI logs.

Fail-open like the Langfuse tracing: if mlflow is not installed or logging
raises, the eval and its pass/fail gate still behave exactly the same.

Where runs go: MLFLOW_TRACKING_URI if set (a tracking server), otherwise a
local ./mlruns folder. CI uploads that folder as a workflow artifact; view
locally with `mlflow ui --backend-store-uri mlruns`.
"""
import os

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT_NAME", "twin-eval")

# MLflow 3.x refuses the plain ./mlruns folder store unless told otherwise.
# CI pins mlflow-skinny 2.x (eval/requirements-tracking.txt), where this is a
# no-op; the default keeps newer local installs working too.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def flatten_metrics(d: dict, prefix: str = "") -> dict[str, float]:
    """Keeps numeric values only (MLflow metrics must be numbers), joins
    nested keys with '.', and turns True/False into 1/0."""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            out[name] = float(value)
        elif isinstance(value, (int, float)):
            out[name] = float(value)
        elif isinstance(value, dict):
            out.update(flatten_metrics(value, f"{name}."))
    return out


def default_params(**extra) -> dict[str, str]:
    params = {
        "git_sha": os.environ.get("GITHUB_SHA", "local")[:12],
        "prompt_source": os.environ.get("PROMPT_SOURCE", "local"),
        "prompt_label": os.environ.get("PROMPT_LABEL", "production"),
        "ci": "true" if os.environ.get("GITHUB_ACTIONS") else "false",
    }
    params.update({k: str(v) for k, v in extra.items()})
    return params


def log_run(run_name: str, params: dict, metrics: dict, artifact_paths: list[str] | None = None,
            tags: dict | None = None) -> str | None:
    """Returns the MLflow run id, or None if tracking is unavailable."""
    try:
        import mlflow

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            for name, value in flatten_metrics(metrics).items():
                mlflow.log_metric(name, value)
            for path in artifact_paths or []:
                if os.path.exists(path):
                    mlflow.log_artifact(path)
            if tags:
                mlflow.set_tags(tags)
            return run.info.run_id
    except Exception as e:  # noqa: BLE001 - tracking must never break an eval
        print(f"mlflow tracking skipped: {type(e).__name__}: {e}")
        return None
