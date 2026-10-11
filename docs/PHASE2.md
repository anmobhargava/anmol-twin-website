Semantic cache analytics
- Each lookup prints a `cache_lookup` JSON line: `outcome` (`hit` / `miss` / `stale` / `empty`) and `similarity`.
- Each cached answer stores what it cost to produce (`gen_ms`, `gen_tokens`); a hit reports `saved_ms` / `saved_tokens` in its `chat_request` log line.
- `/health` (both Lambdas) includes a `cache` block: lookups, hits, misses, stale, `hit_rate`, saved totals. Counters are **per warm container** and reset on cold start.
- CloudWatch metrics in namespace `Twin` (`infra/website/cache_metrics.tf`): `CacheHit`, `CacheMiss`, `CacheSavedMs`, `CacheSavedTokens`. Hit rate = `CacheHit / (CacheHit + CacheMiss)` (metric math).
- Logs Insights: `fields outcome, similarity | filter event = "cache_lookup" | stats count() by outcome`
- Fix included: cache keys used Python's randomized `hash()`; they now use sha256, so the same question maps to the same key across Lambda containers.

## RAGAS
`python eval/run_ragas.py <vector-bucket>` (deps: `eval/requirements-ragas.txt`). Scores faithfulness and context precision for every gradable golden case; context recall for cases that have a `"reference"` field in `golden.json`. Runs from the "RAGAS evaluation" workflow (manual + weekly). It never blocks a deploy.

## MLflow
The eval gate and RAGAS job each log one run (params: git sha, prompt source/label; metrics; report artifact) to `./mlruns` or `MLFLOW_TRACKING_URI`. CI uploads `mlruns` with the report artifact. To browse runs locally: download the CI artifact, then `pip install mlflow && mlflow ui --backend-store-uri mlruns`.

## DVC
`dvc repro` chunks the corpus (no LLM/AWS) into `artifacts/chunks.jsonl` and writes `artifacts/corpus_metrics.json`; `dvc metrics diff` shows what a corpus edit changed. To share artifacts: `dvc remote add -d s3store s3://<regular-s3-bucket>/dvc` then `dvc push`.
