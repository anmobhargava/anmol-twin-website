# infra/website/lambda.tf
#
# Container image deployment (not zip+layers). CPU-only PyTorch alone is
# ~370MB uncompressed -- already over Lambda's 250MB unzipped zip/layer
# limit, before sentence-transformers, faiss-cpu, or scikit-learn are even
# added. Container images support up to 10GB, comfortably covering all of
# this with zero changes to backend/'s actual code (see ../../Dockerfile
# for how the exact backend/+corpus/ sibling layout is preserved so
# lambda_handler.py's existing path logic and dotted handler string keep
# working completely unchanged).

# --- ECR repository for the Lambda's container image ---
resource "aws_ecr_repository" "twin_chat" {
  name                 = "${var.project_name}-chat"
  image_tag_mutability = "MUTABLE"
}

# Without this, ECR keeps every pushed image indefinitely, even after a
# new "latest" push -- a new tag doesn't delete the image it replaced, it
# just moves the tag pointer. Every docker push in this project (and there
# have been many, across debugging sessions and CI runs) accumulates real,
# ongoing storage cost with nothing ever pruning it automatically. This
# caps the repo at the 20 most recent images -- generous headroom for
# rollback if ever needed, while stopping unbounded growth. (Each deploy
# now pushes TWO images -- the chat image and the streaming image, see
# stream.tf -- so 20 keeps the same ~10 deploys of history 10 used to.)
resource "aws_ecr_lifecycle_policy" "twin_chat" {
  repository = aws_ecr_repository.twin_chat.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# Looks up the digest of whatever image is currently tagged "latest" in
# ECR. Referencing the DIGEST (not just the "latest" tag string) in the
# Lambda function below is what makes Terraform correctly detect a new
# image push and redeploy -- if we referenced the "latest" tag directly,
# Terraform would see the same unchanging string on every apply and never
# know a new image had been pushed underneath it.
#
# IMPORTANT -- real chicken-and-egg sequencing, same shape as the
# bootstrap/main-stack split used for Terraform state: this data source
# needs an image to ALREADY exist in ECR before it can succeed, but the
# ECR repo itself doesn't exist until this same `terraform apply` creates
# it. On a truly first-time apply, this means a TWO-STEP process:
#   1. terraform apply -target=aws_ecr_repository.twin_chat
#      (creates just the ECR repo, nothing else)
#   2. Build and push the Docker image, tagged "latest", to that repo
#   3. terraform apply
#      (now the data source finds a real image, and everything else --
#      Lambda, API Gateway, CloudFront, etc. -- can be created)
# Every subsequent code change just needs step 2 + a normal `terraform
# apply` -- the two-step dance is only required once, the very first time.
data "aws_ecr_image" "latest" {
  repository_name = aws_ecr_repository.twin_chat.name
  image_tag       = "latest"

  depends_on = [aws_ecr_repository.twin_chat]
}

# --- IAM role for the Lambda function ---
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Minimum permissions: just CloudWatch Logs. This function calls OUT to the
# Anthropic API (an external HTTPS call needing no IAM permission) and only
# reads files bundled in its own container image -- no S3, no DynamoDB, no
# other AWS service access needed beyond the conversation-log write below.
# (No separate policy is needed for Lambda to pull its OWN container image
# from ECR in the same account -- that access is implicit.)
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Explicit log group with a real retention period -- without this, Lambda
# auto-creates one with NEVER-EXPIRE retention by default, which quietly
# accumulates storage cost forever. 30 days is a reasonable window for a
# recruiter-chatbot-scale project: long enough to debug something that
# happened last week, short enough to not need active cleanup.
resource "aws_cloudwatch_log_group" "twin_chat" {
  name              = "/aws/lambda/${var.project_name}-chat"
  retention_in_days = 30
}

# --- Conversation logging bucket ---
# Private, no public access whatsoever -- these are unauthenticated
# visitors' typed questions, logged so Anmol can see what recruiters
# actually ask and improve the corpus accordingly.
resource "aws_s3_bucket" "conversation_logs" {
  bucket = "${var.project_name}-conversation-logs-anmolbhargava"
}

resource "aws_s3_bucket_public_access_block" "conversation_logs" {
  bucket                  = aws_s3_bucket.conversation_logs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "conversation_logs" {
  bucket = aws_s3_bucket.conversation_logs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# GetObject and ListBucket added after a real deploy failure:
# _load_session() in lambda_handler.py reads session files back (not just
# writes them), but the original policy only ever granted PutObject.
resource "aws_iam_role_policy" "lambda_conversation_log_write" {
  name = "${var.project_name}-conversation-log-write"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.conversation_logs.arn,
        "${aws_s3_bucket.conversation_logs.arn}/*",
      ]
    }]
  })
}

# Permission to call Bedrock's Titan Embeddings model -- vector_store.py
# now calls this via the API instead of running a local sentence-transformers
# model, which removed torch (and its ~370MB+ size, and an unexplained
# cold-start hang) from the deployment entirely. Scoped to exactly the one
# model this function actually calls, not broad Bedrock access.
resource "aws_iam_role_policy" "lambda_bedrock_embeddings" {
  name = "${var.project_name}-bedrock-embeddings"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "bedrock:InvokeModel"
      Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
    }]
  })
}

# --- S3 Vectors bucket ---
# NOT managed by Terraform -- aws_s3vectors_vector_bucket requires AWS
# provider v6.24+, and this project is pinned to v5.x. Bumping to v6 would
# be a major-version upgrade risking breaking changes across every OTHER
# resource in this config. Instead, this bucket is created ONCE, manually,
# via the AWS CLI. var.vector_bucket_name just has to match whatever name
# was used when creating it.
locals {
  vector_bucket_arn = "arn:aws:s3vectors:${var.aws_region}:${data.aws_caller_identity.current.account_id}:bucket/${var.vector_bucket_name}"
}

# GetVectors added after a real deploy failure: AWS's error for a denied
# ListVectors API call explicitly named "s3vectors:GetVectors" as the
# required IAM action -- the operation name and the IAM permission name
# don't always match 1:1.
resource "aws_iam_role_policy" "lambda_s3vectors" {
  name = "${var.project_name}-s3vectors"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3vectors:CreateIndex",
        "s3vectors:PutVectors",
        "s3vectors:QueryVectors",
        "s3vectors:ListVectors",
        "s3vectors:GetVectors",
      ]
      Resource = [
        local.vector_bucket_arn,
        "${local.vector_bucket_arn}/index/*",
      ]
    }]
  })
}

# --- Rate-limit counters ---
# One tiny item per (client, window). On-demand billing (no capacity to
# size, costs pennies at this traffic), and a TTL on `expires_at` so old
# counters delete themselves. See backend/chassis/rate_limit.py.
resource "aws_dynamodb_table" "rate_limits" {
  name         = "${var.project_name}-rate-limits"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"

  attribute {
    name = "pk"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
}

# UpdateItem only, scoped to this one table -- the limiter never reads,
# scans, or deletes.
resource "aws_iam_role_policy" "lambda_rate_limits" {
  name = "${var.project_name}-rate-limits"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["dynamodb:UpdateItem"]
      Resource = aws_dynamodb_table.rate_limits.arn
    }]
  })
}

# Environment shared by the chat function and the streaming function
# (stream.tf). One definition so a new variable can't be added to one and
# forgotten on the other.
locals {
  app_env = {
    ANTHROPIC_API_KEY       = var.anthropic_api_key
    CONVERSATION_LOG_BUCKET = aws_s3_bucket.conversation_logs.id
    VECTOR_BUCKET           = var.vector_bucket_name
    RATE_LIMIT_TABLE        = aws_dynamodb_table.rate_limits.name
    LANGFUSE_PUBLIC_KEY     = var.langfuse_public_key
    LANGFUSE_SECRET_KEY     = var.langfuse_secret_key
    LANGFUSE_HOST           = var.langfuse_host
    GIT_SHA                 = var.git_sha
    PROMPT_SOURCE           = var.prompt_source # "local" (prompts from the image) or "langfuse" (versioned, see scripts/sync_prompts.py)
    PROMPT_LABEL            = var.prompt_label
  }
}

# --- The Lambda function ---
resource "aws_lambda_function" "twin_chat" {
  function_name = "${var.project_name}-chat"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"

  image_uri = "${aws_ecr_repository.twin_chat.repository_url}@${data.aws_ecr_image.latest.image_digest}"

  # 60s/512MB -- covers the real sequential LLM call chain (condense, HyDE,
  # grading, final generation) a single request can make.
  timeout     = 60
  memory_size = 512

  # Every apply that changes the function (new image digest, new GIT_SHA)
  # publishes an immutable numbered VERSION. Versions are what the "live"
  # alias below points at, and what makes instant rollback possible.
  publish = true

  environment {
    variables = local.app_env
  }
}

# --- "live" alias: the stable name API Gateway calls ---
# Traffic goes to whatever version this alias points at, NOT to $LATEST.
# Terraform creates the alias once; after that CI moves it (see
# .github/workflows/deploy.yml: smoke-test the new version, THEN promote)
# and scripts/rollback.sh moves it back. ignore_changes stops later
# `terraform apply` runs from yanking it back to the newest version, which
# would undo a deliberate rollback and bypass the smoke test.
resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.twin_chat.function_name
  function_version = aws_lambda_function.twin_chat.version

  lifecycle {
    ignore_changes = [function_version]
  }
}
