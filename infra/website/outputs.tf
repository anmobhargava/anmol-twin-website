output "api_invoke_url" {
  value       = aws_apigatewayv2_stage.default.invoke_url
  description = "Paste this into frontend/config.js's API_BASE_URL"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.twin_chat.repository_url
  description = "Where to push the Lambda's Docker image -- docker push <this>:latest"
}

output "cloudfront_domain" {
  value       = aws_cloudfront_distribution.frontend.domain_name
  description = "CloudFront's own domain -- useful for testing before DNS propagates"
}

output "frontend_bucket_name" {
  value       = aws_s3_bucket.frontend.id
  description = "Upload frontend/ files here (aws s3 sync frontend/ s3://<this-bucket>/)"
}

output "site_url" {
  value       = "https://${var.domain_name}"
  description = "The final, real URL once DNS has propagated"
}