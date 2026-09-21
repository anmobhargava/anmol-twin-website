# infra/website/log_archive.tf
#
# Exports CloudWatch Logs to S3 for long-term archival/querying, via the
# standard AWS pattern: CloudWatch Logs subscription filter -> Kinesis
# Firehose -> S3. Two IAM roles are needed because two different AWS
# services need to assume a role to do their part: CloudWatch Logs itself
# needs permission to PUSH records into Firehose, and Firehose separately
# needs permission to WRITE those records into S3 -- these are genuinely
# different trust relationships, not one role reused for both.
#
# Reuses the existing conversation_logs bucket (a "cloudwatch-logs/" prefix
# within it) rather than creating a new bucket -- this is pure log archival,
# doesn't need its own dedicated bucket at this project's scale.

data "aws_caller_identity" "current" {}

# --- Firehose's own role: permission to write into S3 ---
resource "aws_iam_role" "firehose_delivery" {
  name = "${var.project_name}-firehose-delivery"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "firehose.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "firehose_s3_write" {
  name = "${var.project_name}-firehose-s3-write"
  role = aws_iam_role.firehose_delivery.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:PutObject",
        "s3:GetBucketLocation",
        "s3:ListBucket",
      ]
      Resource = [
        aws_s3_bucket.conversation_logs.arn,
        "${aws_s3_bucket.conversation_logs.arn}/cloudwatch-logs/*",
      ]
    }]
  })
}

# --- The Firehose delivery stream itself ---
resource "aws_kinesis_firehose_delivery_stream" "cloudwatch_logs" {
  name        = "${var.project_name}-cwl-to-s3"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn   = aws_iam_role.firehose_delivery.arn
    bucket_arn = aws_s3_bucket.conversation_logs.arn
    prefix     = "cloudwatch-logs/"

    # Buffers before writing to S3 -- 5 minutes or 5MB, whichever comes
    # first. At this project's log volume, time will almost always be the
    # trigger, not size -- meaning a delay of up to ~5 minutes between an
    # event happening and it landing in S3, which is fine for archival/
    # analysis, not meant for real-time alerting.
    buffering_interval = 300
    buffering_size      = 5
  }
}

# --- CloudWatch Logs' role: permission to push INTO Firehose ---
# The sts:ExternalId condition (set to this account's own ID) is AWS's
# documented best practice for this specific trust relationship -- it
# prevents a "confused deputy" scenario where some other AWS account
# could otherwise try to assume this role to push data into our Firehose
# stream.
resource "aws_iam_role" "cwl_to_firehose" {
  name = "${var.project_name}-cwl-to-firehose"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "logs.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = {
          "sts:ExternalId" = data.aws_caller_identity.current.account_id
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "cwl_to_firehose_policy" {
  name = "${var.project_name}-cwl-to-firehose-policy"
  role = aws_iam_role.cwl_to_firehose.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
      Resource = aws_kinesis_firehose_delivery_stream.cloudwatch_logs.arn
    }]
  })
}

# --- The actual subscription: every event in twin-website-chat's log
# group gets forwarded to Firehose (and from there, to S3) ---
resource "aws_cloudwatch_log_subscription_filter" "twin_chat_to_firehose" {
  name            = "${var.project_name}-cwl-to-firehose-filter"
  log_group_name  = aws_cloudwatch_log_group.twin_chat.name
  filter_pattern  = ""  # empty pattern = every log event, not just ones matching a filter
  destination_arn = aws_kinesis_firehose_delivery_stream.cloudwatch_logs.arn
  role_arn        = aws_iam_role.cwl_to_firehose.arn
}