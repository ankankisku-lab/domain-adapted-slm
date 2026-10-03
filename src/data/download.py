"""Phase 1: download pinned raw datasets and write a source manifest (URL, revision, sha256, size, record count).

Usage:  python -m src.data.download
"""

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from src.data.sources import SOURCES

RAW_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest/source_manifest.json")


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _count_records(name: str, data: list) -> int:
    # TAT-QA groups several questions under one table/paragraph context.
    return sum(len(ctx["questions"]) for ctx in data) if name == "tatqa" else len(data)


def main() -> None:
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {}}
    for name, src in SOURCES.items():
        entry = {k: src[k] for k in ("role", "homepage", "revision", "licenses", "citation")}
        entry["files"] = {}
        for split, url in src["files"].items():
            path = RAW_DIR / name / f"{split}.json"
            _fetch(url, path)
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["files"][split] = {
                "url": url,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "records": _count_records(name, data),
            }
            print(f"{name}/{split}: {entry['files'][split]['records']} records")
        manifest["sources"][name] = entry

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
