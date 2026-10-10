# Streaming chat

Replies appear word by word instead of all at once.

## How it works
- `frontend/script.js` calls `STREAM_URL + /chat/stream` when `CONFIG.STREAM_URL` is set, otherwise (or if streaming fails before any text appears) the original `/chat`.
- `STREAM_URL` is a **Lambda Function URL** (response-streaming mode) for a second function, `twin-website-stream`: same code, run as a FastAPI web server (`backend/stream_app.py`) through the AWS Lambda Web Adapter (`Dockerfile.stream`). Terraform: `infra/website/stream.tf`.
- Only the final answer streams. Condense / HyDE / grading run first (about 2-4 s), then text flows.
- Wire format: NDJSON lines `delta`, `replace`, `done`, `error` (see the docstring in `stream_app.py`).

## Output guardrail (replace-after-check)
The answer streams, then `check_output` runs on the finished text. If it trips, the page receives a `replace` event and swaps the message for the safe fallback. The unsafe text is never saved to the session.

## Deploy behaviour
`deploy.yml` builds a second image (`stream-latest`), applies Terraform, then health-checks `<stream url>/health` for the current commit SHA. **Only if that passes** is `STREAM_URL` written into `config.js`. A broken streaming function therefore ships the site without streaming (chat keeps working); it never blocks the deploy.

## Turning it off / rollback
- Fastest: set `STREAM_URL: ""` in `frontend/config.js` (and make CI not inject it) -> everyone uses `/chat`.
- The chat function, API Gateway and the `live` alias are untouched by streaming.
- The streaming function has no alias; roll back by redeploying an earlier commit.

## If streaming does not work after deploy
1. CI log, step "Check streaming endpoint" -> warning? The site shipped without streaming.
2. `curl <stream url>/health` returns 403 -> the resource policy is missing the `InvokeFunction ... InvokedViaFunctionUrl` statement. Check `aws lambda get-policy --function-name twin-website-stream` for `FunctionURLInvokeAllowPublicAccess`; add it with `bash scripts/ensure_function_url_permission.sh twin-website-stream us-east-1`.
3. Logs: CloudWatch `/aws/lambda/twin-website-stream` (JSON lines include `"streamed": true`).

## CI user permissions added by this feature
`lambda:CreateFunctionUrlConfig`, `lambda:GetFunctionUrlConfig`, `lambda:UpdateFunctionUrlConfig`, `lambda:DeleteFunctionUrlConfig`, `lambda:AddPermission`, `lambda:RemovePermission`, `lambda:GetPolicy` (plus the existing Lambda/ECR/Logs permissions, which now also apply to the second function).

## Not verified from the build environment
Real streaming through AWS (Function URL + Lambda Web Adapter) could not be exercised from the sandbox. Tested offline: the Python pieces, the NDJSON stream through real uvicorn, and the page's handling of normal / replace / rate-limit / fallback / interrupted streams in a headless browser.
