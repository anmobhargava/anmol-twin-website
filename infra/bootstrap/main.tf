# infra/bootstrap/main.tf
#
# ONE-TIME SETUP for the twin-website project's own Terraform state.
# Self-contained -- does not reuse or reference chassis-agent-01's bootstrap.
# Uses local state itself (same chicken-and-egg reasoning as before: you
# can't use a remote backend that doesn't exist yet to create that backend).

terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "state_bucket_name" {
  description = "Globally-unique S3 bucket name for this project's Terraform state."
  type        = string
  default     = "twin-website-tfstate-anmolbhargava-2026"
}

resource "aws_s3_bucket" "tfstate" {
  bucket = var.state_bucket_name

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tf_lock" {
  name         = "twin-website-tf-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

output "state_bucket_name" {
  value       = aws_s3_bucket.tfstate.id
  description = "Use this exact name in infra/website/backend.tf"
}

output "lock_table_name" {
  value       = aws_dynamodb_table.tf_lock.name
  description = "Use this exact name in infra/website/backend.tf"
}