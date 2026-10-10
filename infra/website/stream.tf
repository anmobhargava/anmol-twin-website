# infra/website/stream.tf
#
# Streaming chat: a SECOND Lambda function (same code, run as a web server --
# see ../../Dockerfile.stream and backend/stream_app.py) exposed through a
# Lambda Function URL in response-streaming mode. The existing chat function,
# its API Gateway route and the `live` alias are untouched: /chat keeps
# working, and the frontend falls back to it whenever streaming is
# unavailable.
#
# Not behind API Gateway on purpose: HTTP APIs can't stream responses.
# Consequences to be aware of:
#   * The URL is public (auth NONE, like the website's API). Abuse protection
#     is the app-level rate limiter + input guardrail, same as /chat.
#   * No alias/gated-promotion for this function. If a bad build ships, the
#     CI health check (deploy.yml) keeps STREAM_URL out of the frontend config
#     and visitors keep using /chat.

data "aws_ecr_image" "stream" {
  repository_name = aws_ecr_repository.twin_chat.name
  image_tag       = "stream-latest"

  depends_on = [aws_ecr_repository.twin_chat]
}

resource "aws_cloudwatch_log_group" "twin_stream" {
  name              = "/aws/lambda/${var.project_name}-stream"
  retention_in_days = 30
}

resource "aws_lambda_function" "twin_stream" {
  function_name = "${var.project_name}-stream"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.twin_chat.repository_url}@${data.aws_ecr_image.stream.image_digest}"

  timeout     = 60
  memory_size = 512

  environment {
    variables = merge(local.app_env, {
      # There is no 29s API Gateway cut-off on a Function URL, so the
      # request-wide LLM time budget can follow the function timeout.
      LLM_REQUEST_BUDGET_SECONDS = "50"
    })
  }

  depends_on = [aws_cloudwatch_log_group.twin_stream]
}

resource "aws_lambda_function_url" "twin_stream" {
  function_name      = aws_lambda_function.twin_stream.function_name
  authorization_type = "NONE"
  invoke_mode        = "RESPONSE_STREAM"

  # Same single allowed origin as the API Gateway CORS config. Preflight
  # (OPTIONS) is answered by Lambda itself from this block.
  cors {
    allow_origins = ["https://${var.domain_name}"]
    allow_methods = ["POST"]
    allow_headers = ["content-type"]
    max_age       = 300
  }
}

# A public function URL needs TWO resource-policy statements (AWS requirement
# for new function URLs): InvokeFunctionUrl, and InvokeFunction restricted to
# calls arriving via the URL.
resource "aws_lambda_permission" "stream_url_public" {
  statement_id           = "FunctionURLAllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.twin_stream.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# The installed AWS provider (~> 5.0) predates the argument that expresses the
# "invoked via function URL" condition, so this statement is added with the
# AWS CLI instead. The script is idempotent (does nothing if the statement
# already exists) and fails loudly on any real error.
resource "terraform_data" "stream_url_invoke_permission" {
  triggers_replace = [aws_lambda_function.twin_stream.function_name]

  provisioner "local-exec" {
    # If this fails (e.g. an AWS CLI too old to know the flag) the stream URL
    # answers 403, CI's health check notices, and the site ships without
    # streaming instead of the whole deploy failing.
    on_failure = continue
    command = "bash ${path.module}/../../scripts/ensure_function_url_permission.sh ${aws_lambda_function.twin_stream.function_name} ${var.aws_region}"
  }

  depends_on = [aws_lambda_permission.stream_url_public]
}

output "stream_url" {
  value       = aws_lambda_function_url.twin_stream.function_url
  description = "Base URL of the streaming endpoint (frontend calls <this>chat/stream)"
}
