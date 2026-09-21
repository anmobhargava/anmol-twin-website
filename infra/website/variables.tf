# infra/website/variables.tf

terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Lambda/DynamoDB/etc. must be created in this region
provider "aws" {
  region = var.aws_region
}

# ACM certificates for CloudFront MUST be issued in us-east-1, regardless of
# where everything else lives -- this is a hard, well-known AWS requirement,
# not a choice. A second provider alias exists specifically for that.
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

variable "aws_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Used as a prefix for resource names"
  type        = string
  default     = "twin-website"
}

variable "domain_name" {
  description = "The custom domain for the twin website, e.g. anmolbhargava.dev"
  type        = string
  # No default on purpose -- this MUST be set explicitly to your real domain
  # before running terraform plan/apply.
}

variable "anthropic_api_key" {
  description = "Anthropic API key for the Lambda function. Marked sensitive so it never prints in plan/apply output or gets logged."
  type        = string
  sensitive   = true
}

variable "vector_bucket_name" {
  description = "Name of the S3 Vectors bucket, created manually via AWS CLI (not Terraform -- see lambda.tf's own comment for why). Must match exactly whatever name was used when creating it."
  type        = string
  default     = "twin-website-vectors"
}