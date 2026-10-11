# infra/website/cache_metrics.tf
#
# Semantic-cache analytics as CloudWatch metrics (namespace "Twin"), derived
# from the `chat_request` JSON lines both Lambdas print. Applied to BOTH log
# groups (REST /chat and streaming), so the numbers cover all traffic.
#
#   CacheHit / CacheMiss     -> hit rate = CacheHit / (CacheHit + CacheMiss)
#                               (CloudWatch metric math, in the console)
#   CacheSavedMs             -> total latency a hit avoided (ms)
#   CacheSavedTokens         -> total LLM tokens a hit avoided
#
# Per-lookup detail (similarity, stale vs miss) is in the `cache_lookup` log
# lines; query it with Logs Insights, e.g.
#   fields outcome, similarity | filter event = "cache_lookup" | stats count() by outcome

locals {
  chat_log_groups = {
    chat   = aws_cloudwatch_log_group.twin_chat.name
    stream = aws_cloudwatch_log_group.twin_stream.name
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_hit" {
  for_each       = local.chat_log_groups
  name           = "${var.project_name}-cache-hit-${each.key}"
  log_group_name = each.value
  pattern        = "{ $.event = \"chat_request\" && $.cache_hit IS TRUE }"
  metric_transformation {
    name          = "CacheHit"
    namespace     = "Twin"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_miss" {
  for_each       = local.chat_log_groups
  name           = "${var.project_name}-cache-miss-${each.key}"
  log_group_name = each.value
  pattern        = "{ $.event = \"chat_request\" && $.cache_hit IS FALSE }"
  metric_transformation {
    name          = "CacheMiss"
    namespace     = "Twin"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_saved_ms" {
  for_each       = local.chat_log_groups
  name           = "${var.project_name}-cache-saved-ms-${each.key}"
  log_group_name = each.value
  pattern        = "{ $.event = \"chat_request\" && $.saved_ms > 0 }"
  metric_transformation {
    name      = "CacheSavedMs"
    namespace = "Twin"
    value     = "$.saved_ms"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_saved_tokens" {
  for_each       = local.chat_log_groups
  name           = "${var.project_name}-cache-saved-tokens-${each.key}"
  log_group_name = each.value
  pattern        = "{ $.event = \"chat_request\" && $.saved_tokens > 0 }"
  metric_transformation {
    name      = "CacheSavedTokens"
    namespace = "Twin"
    value     = "$.saved_tokens"
  }
}
