# infra/website/s3_cloudfront.tf
#
# S3 bucket holds the static frontend files (index.html, styles.css, etc.)
# but is NOT public -- CloudFront is the only thing allowed to read from it,
# via Origin Access Control (OAC, the current recommended approach,
# replacing the older Origin Access Identity). This means the bucket itself
# can stay fully private while still serving content to the world through
# CloudFront's edge network.

resource "aws_s3_bucket" "frontend" {
  bucket = "${var.project_name}-frontend-anmolbhargava"
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket                  = aws_s3_bucket.frontend.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudfront_origin_access_control" "frontend" {
  name                              = "${var.project_name}-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# Bucket policy explicitly allows ONLY this specific CloudFront distribution
# to read objects -- not the public internet, not other AWS accounts.
resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
      Condition = {
        StringEquals = {
          "AWS:SourceArn" = aws_cloudfront_distribution.frontend.arn
        }
      }
    }]
  })
}

# ACM certificate MUST be in us-east-1 for CloudFront, regardless of the
# stack's primary region -- a hard AWS requirement, hence the aliased
# provider from variables.tf.
resource "aws_acm_certificate" "frontend" {
  provider          = aws.us_east_1
  domain_name       = var.domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudfront_distribution" "frontend" {
  enabled             = true
  default_root_object = "index.html"
  aliases             = [var.domain_name]

  origin {
    domain_name              = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id                = "s3-frontend"
    origin_access_control_id = aws_cloudfront_origin_access_control.frontend.id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "s3-frontend"
    viewer_protocol_policy = "redirect-to-https"

    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    # References the VALIDATION resource's certificate_arn (not the raw
    # cert resource directly) -- this is what actually creates the
    # dependency ordering so CloudFront waits for DNS validation to
    # complete before it's created. Referencing aws_acm_certificate.frontend.arn
    # directly here would let Terraform race ahead and try to attach an
    # unvalidated certificate.
    acm_certificate_arn      = aws_acm_certificate_validation.frontend.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
}