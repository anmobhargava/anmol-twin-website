"""
Per-visitor and global rate limiting backed by DynamoDB.

Why this exists: the /chat endpoint is public, unauthenticated, and every
request spends real money (several Anthropic calls plus Bedrock embeddings).
API Gateway's stage-level throttle (infra/website/api_gateway.tf) protects
against bursts, but it is one global bucket -- a single noisy client can
use all of it and lock everyone else out. This adds three counters:

  - per client, per minute   (stops rapid-fire scripts)
  - per client, per hour     (stops slow, sustained abuse)
  - global, per day          (a hard cost ceiling: whatever happens, the
                              Anthropic bill for this endpoint is capped)

State must live outside the Lambda because containers are stateless and
many run concurrently. DynamoDB's atomic ADD makes each counter race-free.
Counters carry a TTL (`expires_at`) so the table cleans itself up.

Client identity is a salted SHA-256 prefix of the source IP, so raw IP
addresses are never written to the table.

FAIL-OPEN, same principle as chassis/tracing.py: if DynamoDB is slow,
throttled or down, requests are ALLOWED. The limiter exists to protect the
product; it must never be able to take the product down. (The API Gateway
throttle and the global daily counter remain as backstops when it works.)
"""

import hashlib
import os
import time
from dataclasses import dataclass


@dataclass
class RateLimitResult:
    allowed: bool
    scope: str | None = None        # which limit tripped: "minute" | "hour" | "daily_global"
    retry_after: int = 0            # seconds until that window resets


class RateLimiter:
    def __init__(
        self,
        table_name: str,
        dynamodb_client=None,
        per_minute: int | None = None,
        per_hour: int | None = None,
        daily_global: int | None = None,
        salt: str | None = None,
        clock=time.time,
    ):
        self.table_name = table_name
        self._client = dynamodb_client
        self.per_minute = per_minute if per_minute is not None else int(os.environ.get("RATE_LIMIT_PER_MINUTE", 6))
        self.per_hour = per_hour if per_hour is not None else int(os.environ.get("RATE_LIMIT_PER_HOUR", 40))
        self.daily_global = daily_global if daily_global is not None else int(os.environ.get("RATE_LIMIT_DAILY_GLOBAL", 1500))
        self.salt = salt if salt is not None else os.environ.get("RATE_LIMIT_SALT", "twin-website")
        self._clock = clock

    @property
    def client(self):
        if self._client is None:
            import boto3  # imported lazily so unit tests and local runs don't need AWS config
            self._client = boto3.client("dynamodb")
        return self._client

    def client_id(self, source_ip: str) -> str:
        return hashlib.sha256(f"{self.salt}:{source_ip}".encode()).hexdigest()[:16]

    def _hit(self, identifier: str, window_seconds: int) -> tuple[int, int]:
        """Atomically increments the counter for the current window and
        returns (count_after_increment, seconds_until_window_resets)."""
        now = int(self._clock())
        window_start = now - (now % window_seconds)
        window_end = window_start + window_seconds
        response = self.client.update_item(
            TableName=self.table_name,
            Key={"pk": {"S": f"{identifier}#{window_seconds}#{window_start}"}},
            UpdateExpression="ADD hits :one SET expires_at = :exp",
            ExpressionAttributeValues={
                ":one": {"N": "1"},
                ":exp": {"N": str(window_end + 300)},  # keep a little past the window, then TTL deletes it
            },
            ReturnValues="UPDATED_NEW",
        )
        return int(response["Attributes"]["hits"]["N"]), max(1, window_end - now)

    def check(self, source_ip: str) -> RateLimitResult:
        try:
            who = self.client_id(source_ip or "unknown")

            # Per-client limits first, global last: a client that is already
            # blocked must not also burn the shared daily budget, or one
            # abusive script could exhaust it for every real visitor.
            count, retry = self._hit(f"ip:{who}", 60)
            if count > self.per_minute:
                return RateLimitResult(False, "minute", retry)

            count, retry = self._hit(f"ip:{who}", 3600)
            if count > self.per_hour:
                return RateLimitResult(False, "hour", retry)

            count, retry = self._hit("global", 86400)
            if count > self.daily_global:
                return RateLimitResult(False, "daily_global", retry)

            return RateLimitResult(True)
        except Exception:
            return RateLimitResult(True)  # fail-open, see module docstring