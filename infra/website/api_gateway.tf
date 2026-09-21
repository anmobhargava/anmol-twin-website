# infra/website/api_gateway.tf
#
# HTTP API (not REST API) -- cheaper, simpler, and the standard choice for
# a single Lambda-backed JSON API like this. CORS is configured NATIVELY
# here, restricted to the actual deployed domain -- not "*" -- per the
# earlier fix: this is what closes off other sites silently using their
# visitors' browsers to call this API. lambda_handler.py itself sets NO
# CORS headers and doesn't handle OPTIONS at all; this config is the single
# source of truth for both.

resource "aws_apigatewayv2_api" "twin_api" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["https://${var.domain_name}"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.twin_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.twin_chat.invoke_arn
  payload_format_version = "2.0" # matches the event shape lambda_handler.py expects (event["requestContext"]["http"]["method"], event["rawPath"])
}

resource "aws_apigatewayv2_route" "chat_route" {
  api_id    = aws_apigatewayv2_api.twin_api.id
  route_key = "POST /chat"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_route" "health_route" {
  api_id    = aws_apigatewayv2_api.twin_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Called once on page load (see script.js's loadHistory()) so a returning
# visitor's prior conversation gets fetched back from S3 instead of
# starting blank. lambda_handler.py already had logic for this path; this
# route was the missing piece connecting API Gateway to it.
resource "aws_apigatewayv2_route" "history_route" {
  api_id    = aws_apigatewayv2_api.twin_api.id
  route_key = "GET /history"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.twin_api.id
  name        = "$default" # a single default stage, no separate /prod or /dev path prefix in the URL
  auto_deploy = true
}

# Grants API Gateway permission to actually invoke this specific Lambda --
# without this, API Gateway can route to it but AWS will refuse the
# invocation at request time with an authorization error.
resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.twin_chat.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.twin_api.execution_arn}/*/*"
}