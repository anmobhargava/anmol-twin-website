# Runbook: roll back a bad deploy

**When:** the site is returning errors / bad answers after a deploy, or an alarm fired (`lambda-errors`, `api-5xx`, `lambda-p95-slow`).

**How it works:** API Gateway calls the Lambda alias `live`. Every deploy publishes a new numbered version; the alias only moves after the smoke test passes. Old versions are kept, so rollback is moving a pointer (seconds, no rebuild).

## Steps

1. See what's live and what you can go back to:
   ```bash
   scripts/rollback.sh
   ```
   The table shows each version and the git commit (`GIT_SHA`) it was built from.
2. Pick the last known-good version and roll back:
   ```bash
   scripts/rollback.sh <version>
   ```
3. Verify (the `version` field is the commit SHA now serving traffic):
   ```bash
   curl -s https://<api-url>/health
   ```
4. Ask one real question on the site.
5. **Fix forward:** push a corrected commit. Do not run `terraform apply` locally to "fix" it.

## Notes
- CI also rolls back automatically if the post-promotion health check fails.
- The alias is protected from Terraform with `ignore_changes`, so a later `apply` will not silently undo a rollback.
- ECR keeps the 10 most recent images (tagged `latest` and by commit SHA).

## Drill (do this once, on purpose)
Roll back to the previous version, confirm `/health` shows the older SHA, then roll forward to the newest version and confirm again. Time it. The goal is that this feels routine.
