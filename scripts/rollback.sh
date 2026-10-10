#!/usr/bin/env bash
# Point the `live` alias at a previous Lambda version. Takes effect in
# seconds; no rebuild, no Terraform.
#
#   scripts/rollback.sh                 # show recent versions and which is live
#   scripts/rollback.sh <version>       # roll live traffic to that version
#
# After a rollback, FIX FORWARD by pushing a corrected commit. Do not run
# `terraform apply` from your laptop to "fix" things: it would republish
# from :latest. (The alias itself is protected by ignore_changes.)
set -euo pipefail

FUNCTION="${FUNCTION_NAME:-twin-website-chat}"
ALIAS="live"

current=$(aws lambda get-alias --function-name "$FUNCTION" --name "$ALIAS" --query FunctionVersion --output text)

if [[ $# -eq 0 ]]; then
  echo "Function: $FUNCTION   live version: $current"
  echo
  echo "Recent versions (version, deployed commit, last modified):"
  aws lambda list-versions-by-function --function-name "$FUNCTION" \
    --query 'Versions[?Version!=`$LATEST`].[Version,Environment.Variables.GIT_SHA,LastModified]' \
    --output table
  exit 0
fi

target="$1"
if [[ "$target" == "$current" ]]; then
  echo "Version $target is already live. Nothing to do."
  exit 0
fi

echo "Rolling $FUNCTION/$ALIAS: version $current -> $target"
aws lambda update-alias --function-name "$FUNCTION" --name "$ALIAS" --function-version "$target" \
  --query '[Name,FunctionVersion]' --output text
echo "Done. Verify: curl -s \$API_URL/health   (the version field shows the commit SHA)"
