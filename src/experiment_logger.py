"""Experiment log: one JSON file per run under experiments/runs/, no paid tracking service.

One file per run (not a shared append-only file) so runs logged on Colab and locally never conflict when both are
pushed to git.
"""

import json
import time
import uuid
from pathlib import Path

RUNS_DIR = Path("experiments/runs")


def log_run(phase: str, config: dict, metrics: dict, runs_dir: Path = RUNS_DIR) -> str:
    """Write a run record and return its id."""
    run_id = f"{phase}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "phase": phase, "config": config, "metrics": metrics}
    (runs_dir / f"{run_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return run_id


def load_runs(phase: str | None = None, runs_dir: Path = RUNS_DIR) -> list[dict]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]
    return [r for r in runs if phase is None or r["phase"] == phase]
