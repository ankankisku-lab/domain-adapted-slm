"""Background watcher for a running SFT job on Colab: reports progress to GitHub and runs the follow-up eval.

Started once, detached from the notebook, so VS Code disconnects don't matter:
  nohup python -m src.watch_sft --tag sft_r16 > logs/watch_sft_r16.log 2>&1 &

Every --interval seconds it writes results/<tag>_progress.json (latest checkpoint step, train/eval loss history,
best checkpoint, whether the trainer is alive) and pushes it (src/colab_sync.py; needs GIT_PUSH_URL). When the
trainer exits:
  - metrics file present -> push training metrics, run the test-set eval (resumable, pushes as it goes), finish
  - metrics file missing -> report "trainer_exited_without_metrics" and stop (crash or kill)
"""

import argparse
import glob
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src.colab_sync import sync


def trainer_alive(tag: str) -> bool:
    return subprocess.run(["pgrep", "-f", f"src.train.sft --tag {tag}"], capture_output=True).returncode == 0


def checkpoint_state(tag: str) -> dict | None:
    cks = sorted(glob.glob(f"outputs/{tag}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
    if not cks:
        return None
    st = json.loads(Path(cks[-1], "trainer_state.json").read_text())
    return {
        "checkpoint": cks[-1], "step": st["global_step"], "max_steps": st["max_steps"],
        "eval_loss": [{"step": h["step"], "eval_loss": round(h["eval_loss"], 4)}
                      for h in st["log_history"] if "eval_loss" in h],
        "train_loss": [{"step": h["step"], "loss": h["loss"]} for h in st["log_history"] if "loss" in h],
        "best_checkpoint": st.get("best_model_checkpoint"), "best_eval_loss": st.get("best_metric"),
    }


def report(tag: str, stage: str, extra: dict | None = None) -> None:
    path = Path(f"results/{tag}_progress.json")
    path.parent.mkdir(exist_ok=True)
    body = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage,
            "trainer_alive": trainer_alive(tag), "checkpoint_state": checkpoint_state(tag), **(extra or {})}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    step = (body["checkpoint_state"] or {}).get("step", 0)
    sync([str(path)], f"WIP {tag}: {stage}, checkpoint step {step}")
    print(f"[{body['updated_utc']}] {stage} step={step} alive={body['trainer_alive']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="sft_r16")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    args = ap.parse_args()
    metrics = Path(f"results/{args.tag}_train_metrics.json")

    while trainer_alive(args.tag):
        report(args.tag, "training")
        time.sleep(args.interval)

    if not metrics.exists():
        report(args.tag, "trainer_exited_without_metrics")
        return
    report(args.tag, "training_done")
    sync([str(metrics), "experiments/runs"], f"SFT {args.tag}: training metrics")

    report(args.tag, "evaluating")
    rc = subprocess.run(["python", "-m", "src.eval.run_eval", "--model", f"outputs/{args.tag}/adapter",
                         "--tag", args.tag, "--split", "test", "--batch-size", str(args.eval_batch_size),
                         "--sync-every", "8"]).returncode
    report(args.tag, "done" if rc == 0 else f"eval_failed_rc_{rc}")


if __name__ == "__main__":
    main()
