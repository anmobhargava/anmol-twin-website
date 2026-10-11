"""DVC stage: chunk the corpus (no LLM, no AWS) and write a manifest.

Outputs
  artifacts/chunks.jsonl         every leaf chunk, exactly as build_index.py sees it
  artifacts/corpus_metrics.json  small, diff-able numbers (DVC "metrics")

Why this exists: the live index is rebuilt on every deploy, but nothing
recorded WHICH corpus + chunking produced a given index. `dvc repro` makes
that a tracked, reproducible stage, and `dvc metrics diff` shows what a
corpus edit changed (files, chunks, characters) before it is deployed.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "rag"))

from backend.rag.chunking import load_corpus  # noqa: E402


def main():
    corpus_dir = ROOT / "corpus"
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    chunks = load_corpus(corpus_dir)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "source": c.source, "section": c.section,
                                "chars": len(c.text), "is_child": bool(c.parent_text),
                                "text": c.text}, ensure_ascii=False) + "\n")

    files = sorted(corpus_dir.glob("*.md"))
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.name.encode())
        digest.update(p.read_bytes())

    metrics = {
        "corpus_files": len(files),
        "chunks": len(chunks),
        "child_chunks": sum(1 for c in chunks if c.parent_text),
        "total_chars": sum(len(c.text) for c in chunks),
        "avg_chunk_chars": round(sum(len(c.text) for c in chunks) / max(len(chunks), 1)),
        "corpus_sha256": digest.hexdigest()[:16],
    }
    with open(out_dir / "corpus_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
