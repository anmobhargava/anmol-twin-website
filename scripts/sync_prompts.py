#!/usr/bin/env python3
"""Push the prompts defined in backend/chassis/prompts.py to Langfuse.

For each prompt: if its text differs from the latest Langfuse version, create
a new version (labelled "latest"); if identical, do nothing. With --promote
the version matching the repo text also gets the "production" label, which is
what the live Lambda fetches when PROMPT_SOURCE=langfuse.

  python scripts/sync_prompts.py            # create new versions only
  python scripts/sync_prompts.py --promote  # ...and make them live
  python scripts/sync_prompts.py --dry-run  # show what would happen

Needs LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST in the env.
Rolling back = in the Langfuse UI move the "production" label to an older
version (takes effect within PROMPT_CACHE_SECONDS, no deploy).
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from chassis.prompts import PROMPTS, langfuse_name, to_langfuse  # noqa: E402


def plan(existing: dict[str, str | None]) -> dict[str, str]:
    """name -> 'create' | 'unchanged' given each prompt's latest remote text
    (None = does not exist yet). Pure function so it can be unit tested."""
    out = {}
    for name, template in PROMPTS.items():
        remote = existing.get(name)
        out[name] = "unchanged" if remote == to_langfuse(template) else "create"
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from langfuse import Langfuse
    lf = Langfuse()
    existing: dict[str, str | None] = {}
    versions: dict[str, object] = {}
    for name in PROMPTS:
        try:
            p = lf.get_prompt(langfuse_name(name), label="latest", cache_ttl_seconds=0, max_retries=0)
            existing[name] = p.prompt if isinstance(p.prompt, str) else None
            versions[name] = p
        except Exception:
            existing[name] = None

    actions = plan(existing)
    for name, action in actions.items():
        print(f"{langfuse_name(name):22s} {action}")
        if args.dry_run:
            continue
        labels = ["latest"] + (["production"] if args.promote else [])
        if action == "create":
            lf.create_prompt(name=langfuse_name(name), prompt=to_langfuse(PROMPTS[name]), labels=labels,
                             type="text", commit_message=f"repo commit {_git_sha()}")
        elif args.promote:
            lf.update_prompt(name=langfuse_name(name), version=versions[name].version,
                             new_labels=["production"])
    lf.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
