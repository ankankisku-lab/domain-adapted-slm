"""Run a chain of GPU steps unattended on Colab, reporting to GitHub. Launch once, detached:
  nohup python -m src.jobs align > logs/job_align.log 2>&1 &

Each step is a command plus a "done" check, so rerunning the job after a crash or a new VM skips finished steps.
Before starting, the job waits until no other training/eval/sampling process holds the GPU. Progress goes to
results/job_<name>_progress.json (step statuses + the last lines of the running step's log) and is pushed on every
step change and every few minutes (src/colab_sync.py).
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.colab_sync import sync

# Anything that holds (or is about to launch work on) the GPU. The SFT watcher counts: it starts the SFT eval.
GPU_PATTERN = r"src\.train\.|src\.eval\.run_eval|src\.pref\.sample|src\.watch_sft"
REPORT_EVERY_S = 300
NOISE = ("Warning", "warn(", "return original", "is deprecated", "━━", "Failed to load /usr", "it/s]", "seem to have been set")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def line_count(path: str) -> int:
    p = Path(path)
    return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0


def hf_user() -> str:
    from huggingface_hub import whoami
    return whoami()["name"]


def fetch_adapter(repo: str, local_dir: str) -> list[str]:
    """Command to download a final adapter from the HF Hub (for a fresh VM)."""
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo!r}, local_dir={local_dir!r}, ignore_patterns=['last-checkpoint/*'])")
    return [sys.executable, "-c", code]


def eval_step(tag: str, model: str, split: str, batch: int = 16) -> dict:
    return {"name": f"eval_{tag}_{split}",
            "cmd": [sys.executable, "-m", "src.eval.run_eval", "--model", model, "--tag", tag, "--split", split,
                    "--batch-size", str(batch), "--sync-every", "8"],
            "done": lambda t=tag, s=split: Path(f"results/{t}_{s}_metrics.json").exists(),
            "sync": []}  # run_eval pushes its own results


def job_align() -> list[dict]:
    user = hf_user()
    sft, orpo = "outputs/sft_r16/adapter", "outputs/orpo_r16/adapter"
    return [
        {"name": "fetch_sft_adapter", "cmd": fetch_adapter(f"{user}/finqa-llama32-3b-sft-r16", sft),
         "done": lambda: Path(sft, "adapter_model.safetensors").exists(), "sync": []},
        # Resumes from the predictions already pushed to GitHub if an earlier machine was recycled mid-run.
        eval_step("sft_r16", sft, "test"),
        {"name": "sample_pref_prompts",
         "cmd": [sys.executable, "-m", "src.pref.sample", "--model", sft, "--tag", "sft_r16", "--k", "4",
                 "--temperature", "0.8", "--prompts-per-batch", "4", "--sync-every", "5"],
         "done": lambda: line_count("results/sft_r16_pref_prompts_samples.jsonl") >= 1212,
         "sync": ["results/sft_r16_pref_prompts_samples.jsonl"]},
        {"name": "build_pairs",
         "cmd": [sys.executable, "-m", "src.pref.build_pairs", "--samples", "results/sft_r16_pref_prompts_samples.jsonl"],
         "done": lambda: Path("data/final/pref_pairs.jsonl").exists(),
         "sync": ["data/final/pref_pairs.jsonl", "data/manifest/pref_pairs_report.json"]},
        {"name": "train_orpo",
         "cmd": [sys.executable, "-m", "src.train.orpo", "--tag", "orpo_r16", "--sft-adapter", sft,
                 "--hub-repo", f"{user}/finqa-llama32-3b-orpo-r16", "--resume"],
         "done": lambda: Path("results/orpo_r16_train_metrics.json").exists(),
         "sync": ["results/orpo_r16_train_metrics.json", "experiments/runs"]},
        eval_step("orpo_r16", orpo, "test"),
        # TAT-QA runs locally on the laptop with llama.cpp (src/eval/run_eval_gguf.py) for all three models:
        # Colab Pro ends sessions after ~90 min without user interaction, so Colab time is kept to the minimum.
    ]


JOBS = {"align": job_align}


class Reporter:
    def __init__(self, job: str, steps: list[dict]):
        self.path = Path(f"results/job_{job}_progress.json")
        self.job = job
        self.status = {s["name"]: {"status": "pending"} for s in steps}
        self.last_push = 0.0

    def update(self, step: str | None, log: Path | None = None, force: bool = False, **fields) -> None:
        if step:
            self.status[step].update(fields)
        tail = []
        if log and log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = [l[-200:] for l in lines if l.strip() and not any(n in l for n in NOISE)][-6:]
        body = {"job": self.job, "updated_utc": now(), "current": step, "steps": self.status, "log_tail": tail}
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(body, indent=2), encoding="utf-8")
        if force or time.time() - self.last_push > REPORT_EVERY_S:
            sync([str(self.path)], f"WIP job {self.job}: {step or 'waiting'} {self.status.get(step, {}).get('status', '')}")
            self.last_push = time.time()


def gpu_busy() -> bool:
    return subprocess.run(["pgrep", "-f", GPU_PATTERN], capture_output=True).returncode == 0


def main(job: str) -> None:
    steps = JOBS[job]()
    rep = Reporter(job, steps)
    for s in steps:
        if s["done"]():
            rep.update(s["name"], status="done (already)")
    rep.update(None, force=True)
    while gpu_busy():
        rep.update(None)
        time.sleep(60)

    Path("logs").mkdir(exist_ok=True)
    for s in steps:
        if s["done"]():
            continue
        log = Path(f"logs/{job}_{s['name']}.log")
        rep.update(s["name"], log, force=True, status="running", started=now())
        with log.open("a", encoding="utf-8") as f:
            proc = subprocess.Popen(s["cmd"], stdout=f, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(30)
                rep.update(s["name"], log)
        ok = proc.returncode == 0 and s["done"]()
        rep.update(s["name"], log, force=True, status="done" if ok else "failed", finished=now(),
                   returncode=proc.returncode)
        if s["sync"] and ok:
            sync(s["sync"], f"{job}: {s['name']} done")
        if not ok:
            return  # stop the chain; the report shows which step failed and its last log lines
    rep.update(None, force=True)


if __name__ == "__main__":
    main(sys.argv[1])
