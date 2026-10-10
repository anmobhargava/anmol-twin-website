#!/usr/bin/env bash
# Adds the "InvokeFunction, only when invoked via a function URL" statement to
# a function's resource policy, unless it is already there.
#
# Why this exists: AWS requires BOTH lambda:InvokeFunctionUrl and
# lambda:InvokeFunction (with the InvokedViaFunctionUrl condition) on public
# function URLs, and the pinned Terraform AWS provider (~> 5.0, locked at
# 5.100.0) has no argument for that condition.
#
# Usage: ensure_function_url_permission.sh <function-name> <region>
# Needs AWS CLI v2 recent enough to know --invoked-via-function-url.
set -euo pipefail

FN="${1:?function name}"
REGION="${2:?region}"
SID="FunctionURLInvokeAllowPublicAccess"

policy=$(aws lambda get-policy --function-name "$FN" --region "$REGION" --query Policy --output text 2>/dev/null || true)
if [ -n "$policy" ] && [ "$policy" != "None" ] && echo "$policy" | grep -q "$SID"; then
  echo "Permission $SID already present on $FN."
  exit 0
fi

aws lambda add-permission \
  --function-name "$FN" \
  --region "$REGION" \
  --statement-id "$SID" \
  --action lambda:InvokeFunction \
  --principal '*' \
  --invoked-via-function-url
echo "Added $SID to $FN."
