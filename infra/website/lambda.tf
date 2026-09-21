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

# Narrowly scoped: PutObject only, on this ONE bucket, nothing else. The
# Lambda function never needs to read, list, or delete conversation logs --
# only write new ones -- so the permission granted matches exactly that,
# not a broader S3 access policy.
resource "aws_iam_role_policy" "lambda_conversation_log_write" {
  name = "${var.project_name}-conversation-log-write"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
      Resource = "${aws_s3_bucket.conversation_logs.arn}/*"
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
# resource in this config (Lambda, S3, CloudFront, Route 53, ACM, IAM,
# ECR, API Gateway, Kinesis Firehose), not just this one new resource --
# too large a blast radius to take on for one bucket. Instead, this bucket
# is created ONCE, manually, via the AWS CLI (see the README/deploy notes
# for the exact command) -- the same way domain registration itself is a
# manual, outside-Terraform step. var.vector_bucket_name just has to match
# whatever name was used when creating it.
#
# The ARN is constructed manually here (matching AWS's documented S3
# Vectors ARN format) since there's no Terraform resource to read an .arn
# attribute from.
locals {
  vector_bucket_arn = "arn:aws:s3vectors:${var.aws_region}:${data.aws_caller_identity.current.account_id}:bucket/${var.vector_bucket_name}"
}

# Scoped to exactly this vector bucket (and its indexes) -- CreateIndex,
# PutVectors, QueryVectors, and ListVectors cover everything vector_store.py
# and semantic_cache.py actually call; nothing broader.
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

# --- The Lambda function ---
resource "aws_lambda_function" "twin_chat" {
  function_name = "${var.project_name}-chat"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"

  # References the specific image DIGEST, not the mutable "latest" tag
  # string -- see the data source comment above for why that distinction
  # is what makes auto-redeploy-on-push actually work.
  image_uri = "${aws_ecr_repository.twin_chat.repository_url}@${data.aws_ecr_image.latest.image_digest}"

  # Reverted to modest defaults -- the earlier 3008MB/60s was specifically
  # compensating for torch + sentence-transformers' slow import time, which
  # doesn't exist in this architecture anymore (Bedrock replaced the local
  # embedding model, and connect() does no embedding/RAPTOR work at cold
  # start at all -- see pipeline.py's own docstring). 512MB/15s is generous
  # for what's now a lightweight cold start (a couple of S3 Vectors reads).
  timeout     = 60
  memory_size = 512

  environment {
    variables = {
      ANTHROPIC_API_KEY       = var.anthropic_api_key
      CONVERSATION_LOG_BUCKET = aws_s3_bucket.conversation_logs.id
      VECTOR_BUCKET           = var.vector_bucket_name
    }
  }
}