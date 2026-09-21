# infra/website/backend.tf
#
# Points at the bucket/table created by infra/bootstrap/main.tf.
# Same Terraform limitation as before: backend blocks can't use variables,
# so this has to be a literal value matching bootstrap's output exactly.

terraform {
  backend "s3" {
    bucket         = "twin-website-tfstate-anmolbhargava-2026"
    key            = "website/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "twin-website-tf-lock"
    encrypt        = true
  }
}