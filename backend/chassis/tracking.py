"""
Run-level experiment tracking, via MLflow.

The distinction this chassis draws, deliberately: tracking.py answers "what
happened in this RUN overall" (an extraction pass over the corpus, a full
evaluation sweep, a retrieval-tuning experiment) -- params going in,
metrics coming out, comparable across runs over time. tracing.py (the
other chassis module) answers a different question: "what happened inside
EVERY INDIVIDUAL LLM call within that run." Conflating the two loses the
thing that makes each useful: MLflow's UI is built for comparing runs
side by side (did prompt v2 beat prompt v1?); Langfuse's is built for
drilling into one specific call that went wrong.

Every agent in the portfolio imports from here rather than calling mlflow
directly -- so if the tracking server ever moves, or the experiment-naming
convention changes, it changes in ONE place, not once per agent.
"""

import os
from contextlib import contextmanager

import mlflow

# Defaults to a local SQLite file, not a bare local directory -- verified
# directly against the installed mlflow==3.16.1: the plain filesystem
# backend (the old "./mlruns" default) is now in maintenance mode and
# raises on use unless explicitly opted back into. SQLite is MLflow's own
# recommended minimum for anything beyond throwaway local testing, so
# this default is the genuinely correct choice, not just a workaround.
# Point this at a real tracking SERVER (http://...) once one exists,
# shared across every agent in the portfolio.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _configure():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@contextmanager
def tracked_run(agent_name: str, run_name: str, params: dict | None = None):
    """Wraps one discrete unit of work (an extraction pass, an eval sweep,
    a retrieval-tuning experiment) as an MLflow run.

    agent_name becomes the MLflow EXPERIMENT (e.g. "agent-25-graphrag") --
    this is the grouping that lets you compare runs of the SAME kind of
    work against each other over time, without agent-25's runs cluttering
    agent-26's comparison view.

    run_name is this specific run's label (e.g. "extraction-2026-09-22").

    params are logged immediately, at the start -- these are the INPUTS
    that make a run reproducible (which prompt version, which model,
    which corpus version). Usage:

        with tracked_run("agent-25-graphrag", "extraction-run",
                          params={"prompt_version": "v2", "model": "claude-sonnet-4-6"}) as run:
            entities, relations = extract_from_corpus(corpus)
            log_metrics({"entities_extracted": len(entities), "relations_extracted": len(relations)})
    """
    _configure()
    mlflow.set_experiment(agent_name)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics: dict[str, float]):
    """Logs OUTCOME numbers for the currently-active run (must be called
    inside a `with tracked_run(...):` block) -- e.g. entities_extracted,
    extraction_duration_seconds, faithfulness_score. These are what you
    actually compare across runs to answer "did this change help?" """
    mlflow.log_metrics(metrics)


def log_artifact(local_path: str):
    """Attaches a file to the current run -- e.g. a snapshot of the
    extracted entities/relations JSON, so a specific run's exact output is
    inspectable later, not just its summary metrics."""
    mlflow.log_artifact(local_path)