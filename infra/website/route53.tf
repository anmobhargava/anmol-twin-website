# infra/website/route53.tf
#
# Assumes the domain is ALREADY REGISTERED (in Route 53 or transferred to
# use Route 53 for DNS) -- this reads the existing hosted zone rather than
# creating one, since domain registration itself isn't something Terraform
# manages the same way as these other resources.

data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

# ACM certificates issued with DNS validation need a specific CNAME record
# proving you control the domain, before AWS will actually issue the cert.
# This creates that record automatically from ACM's own validation options,
# rather than you copy-pasting it manually from the console.
resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.frontend.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }

  zone_id         = data.aws_route53_zone.main.zone_id
  name            = each.value.name
  type            = each.value.type
  records         = [each.value.record]
  ttl             = 60
  allow_overwrite = true
}

# Waits for ACM to actually confirm validation succeeded before anything
# downstream (CloudFront) tries to use the certificate -- without this,
# there's a race condition where CloudFront could be created referencing a
# certificate that isn't valid yet.
resource "aws_acm_certificate_validation" "frontend" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.frontend.arn
  validation_record_fqdns = [for r in aws_route53_record.cert_validation : r.fqdn]
}

# The actual DNS record pointing your domain at CloudFront. An ALIAS record
# (Route 53's own extension, not standard DNS) rather than a CNAME -- this
# is required at the apex/root domain, since standard CNAMEs aren't allowed
# there, and it's also just more efficient than a CNAME even at subdomains.
resource "aws_route53_record" "frontend" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"

  alias {
    name                   = aws_cloudfront_distribution.frontend.domain_name
    zone_id                = aws_cloudfront_distribution.frontend.hosted_zone_id
    evaluate_target_health = false
  }
}