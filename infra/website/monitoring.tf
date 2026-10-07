# infra/website/monitoring.tf
#
# Alarms + budget. Everything notifies one SNS topic. Design rule: alarms
# fire on SYMPTOMS a visitor or the bill would feel (errors, slowness,
# fallbacks, spend), not on every internal event.

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- Log-derived metrics (from the JSON lines the handler/llm layer print) ---

resource "aws_cloudwatch_log_metric_filter" "guardrail_blocked" {
  name           = "${var.project_name}-guardrail-blocked"
  log_group_name = aws_cloudwatch_log_group.twin_chat.name
  pattern        = "{ $.event = \"guardrail_blocked\" }"
  metric_transformation {
    name          = "GuardrailBlocked"
    namespace     = "Twin"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "rate_limited" {
  name           = "${var.project_name}-rate-limited"
  log_group_name = aws_cloudwatch_log_group.twin_chat.name
  pattern        = "{ $.event = \"rate_limited\" }"
  metric_transformation {
    name          = "RateLimited"
    namespace     = "Twin"
    value         = "1"
    default_value = "0"
  }
}

# llm_fallback is emitted via Python logging (Lambda prefixes the line with
# level/request id), so it is not pure JSON -- match on the plain term.
resource "aws_cloudwatch_log_metric_filter" "llm_fallback" {
  name           = "${var.project_name}-llm-fallback"
  log_group_name = aws_cloudwatch_log_group.twin_chat.name
  pattern        = "\"llm_fallback\""
  metric_transformation {
    name          = "LlmFallback"
    namespace     = "Twin"
    value         = "1"
    default_value = "0"
  }
}

# --- Alarms ---

resource "aws_cloudwatch_metric_alarm" "lambda_errors" {
  alarm_name          = "${var.project_name}-lambda-errors"
  alarm_description   = "Lambda raised unhandled errors."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.twin_chat.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

# API Gateway cuts at ~29s; p95 above 20s means visitors are close to 503s.
resource "aws_cloudwatch_metric_alarm" "lambda_slow" {
  alarm_name          = "${var.project_name}-lambda-p95-slow"
  alarm_description   = "p95 Lambda duration above 20s (gateway timeout is ~29s)."
  namespace           = "AWS/Lambda"
  metric_name         = "Duration"
  dimensions          = { FunctionName = aws_lambda_function.twin_chat.function_name }
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 2
  threshold           = 20000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  alarm_name          = "${var.project_name}-api-5xx"
  alarm_description   = "Visitors are receiving 5xx responses from the API."
  namespace           = "AWS/ApiGateway"
  metric_name         = "5xx"
  dimensions          = { ApiId = aws_apigatewayv2_api.twin_api.id }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "llm_fallback" {
  alarm_name          = "${var.project_name}-llm-fallback"
  alarm_description   = "Sonnet is failing and requests are falling back to Haiku."
  namespace           = "Twin"
  metric_name         = "LlmFallback"
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Many rate-limit hits in an hour = someone hammering the endpoint.
resource "aws_cloudwatch_metric_alarm" "abuse" {
  alarm_name          = "${var.project_name}-rate-limit-spike"
  alarm_description   = "Unusual volume of rate-limited requests."
  namespace           = "Twin"
  metric_name         = "RateLimited"
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 50
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# --- Cost ---

resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [1]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 80
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }

  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [1]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 100
      threshold_type             = "PERCENTAGE"
      notification_type          = "FORECASTED"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
