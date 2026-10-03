"""
Corpus and artifact versioning, via DVC.

The problem this solves: build_index.py's output (the S3 Vectors index) is
entirely DERIVED from corpus/ -- but nothing today records WHICH version of
corpus/ produced a given index. If retrieval quality changes, "did the
corpus change or did the prompt change?" is currently unanswerable except
by memory. DVC fixes this by giving every corpus snapshot a real,
content-addressed identity (a hash), separate from git's own history --
git tracks CODE changes, DVC tracks DATA changes, and conflating the two
(committing a multi-MB corpus directly into git) is exactly what DVC
exists to avoid.

Scope, stated honestly: this wraps DVC's basic add/commit-tracking
workflow (a corpus snapshot gets a version, retrievable later) -- NOT full
DVC pipeline DAGs (dvc.yaml stages with declared dependencies and
`dvc repro`-style reproducible re-runs). That's a real, deeper DVC
capability this chassis doesn't use yet; add it here if a later agent's
build genuinely needs declared multi-stage reproducibility, rather than
bolting it on speculatively now.
"""

import subprocess
from pathlib import Path

from dvc.repo import Repo


def version_corpus(corpus_dir: str) -> str:
    """Snapshots the current state of corpus_dir as a new DVC-tracked
    version. Returns the resulting .dvc file's content hash (DVC's
    equivalent of a git commit SHA for data) -- this is the identifier to
    log as an MLflow param (see tracking.py's `params` argument) so a
    build_index.py run and the exact corpus state that produced it stay
    linked, even after the corpus changes again later.

    Requires the directory to already be inside an initialized DVC repo
    (`dvc init`, a one-time setup step per project -- not something this
    function does itself, since re-initializing on every call would be
    wrong for a directory that's already tracked).
    """
    repo = Repo(str(Path(corpus_dir).parent))
    stages = repo.add(corpus_dir)

    # The .dvc file DVC writes alongside the tracked directory contains
    # the actual content hash -- reading it back out is how we get a
    # concrete version identifier to return, rather than just "it worked."
    dvc_file = Path(f"{corpus_dir}.dvc")
    content = dvc_file.read_text()
    # DVC's .dvc files are YAML; the md5 hash sits under outs[0].md5 --
    # parsed directly here rather than pulling in a yaml dependency for
    # one field, since the format is stable and simple enough to grep.
    for line in content.splitlines():
        if "md5:" in line:
            return line.split("md5:")[1].strip()

    raise RuntimeError(f"Could not find content hash in {dvc_file} -- DVC's .dvc file format may have changed; re-verify against the installed dvc version.")


def push_corpus():
    """Pushes the versioned corpus data to DVC's configured remote storage
    (an S3 bucket, typically -- configured separately via `dvc remote add`,
    not something this chassis manages) -- without this, version_corpus()
    only records the version LOCALLY; anyone else (or CI) checking out this
    commit would have the .dvc pointer file but not the actual data behind
    it."""
    subprocess.run(["dvc", "push"], check=True)


def corpus_version_at(repo_dir: str, dvc_file_relpath: str = "corpus.dvc", commit_ref: str = "HEAD") -> str | None:
    """Reads back the content hash a PAST commit's .dvc file recorded --
    e.g. corpus_version_at(".", commit_ref="HEAD~3") to answer "what corpus
    version was live 3 commits ago." Returns None if that commit's .dvc
    file didn't exist or couldn't be read, rather than raising -- a missing
    historical version is a legitimate, expected answer for old-enough
    history, not an error condition.

    repo_dir is REQUIRED, not inferred from the caller's current working
    directory -- an earlier version of this function assumed CWD was
    always the right git repo, which silently returned None for every
    call made from anywhere else (caught by real testing, not by
    inspection). dvc_file_relpath defaults to "corpus.dvc" but is
    overridable, since not every agent's tracked file will have that
    exact name/location.
    """
    try:
        content = subprocess.run(
            ["git", "show", f"{commit_ref}:{dvc_file_relpath}"],
            capture_output=True, text=True, check=True, cwd=repo_dir,
        ).stdout
    except subprocess.CalledProcessError:
        return None

    for line in content.splitlines():
        if "md5:" in line:
            return line.split("md5:")[1].strip()
    return None