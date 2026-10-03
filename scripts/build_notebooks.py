"""Generate the Colab notebooks from one definition, so they share a single tested setup cell.

Run after editing:  python scripts/build_notebooks.py
Every code cell is parsed before anything is written (shell/magic lines excepted).
"""

import ast
import json
from pathlib import Path

NB_DIR = Path("notebooks")

SETUP = '''
import getpass, os, shutil, subprocess

REPO = "ankankisku-lab/domain-adapted-slm"                       # public: code and data (no token needed to clone)
SYNC_REPO = "ankankisku-lab/domain-adapted-slm-private-archive"  # private: work-in-progress results from Colab
WORKDIR = "/content/domain-adapted-slm"
token = getpass.getpass("GitHub token (with access to the private results repo): ")
SYNC_URL = f"https://x-access-token:{token}@github.com/{SYNC_REPO}.git"  # passed per command, never stored
os.environ["GIT_PUSH_URL"] = SYNC_URL  # src/colab_sync.py pushes progress here; memory only

def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:  # never echo the command: it may contain the token
        raise RuntimeError((r.stderr or r.stdout).replace(token, "***"))
    return r.stdout

if not os.path.exists(WORKDIR):
    git("clone", "-q", f"https://github.com/{REPO}.git", WORKDIR)
os.chdir(WORKDIR)
git("pull", "-q", "--ff-only", "origin", "main")
print(git("log", "--oneline", "-3"))

# Bring in results already pushed by earlier sessions, so resumable steps continue instead of starting over.
shutil.rmtree(".colab_sync", ignore_errors=True)
git("clone", "-q", "--depth", "1", SYNC_URL, ".colab_sync")
for sub in ("results", "experiments/runs"):
    src = os.path.join(".colab_sync", sub)
    if os.path.isdir(src):
        shutil.copytree(src, sub, dirs_exist_ok=True)
for f in ("data/final/pref_pairs.jsonl", "data/manifest/pref_pairs_report.json"):
    if os.path.exists(os.path.join(".colab_sync", f)):
        shutil.copy2(os.path.join(".colab_sync", f), f)
print("restored saved results:", len(os.listdir("results")), "files")
'''

INSTALL = "%pip install -q -r requirements-colab.txt"


def push(files: list[str], subject: str) -> str:
    return f"""
from src.colab_sync import sync
print("saved to the private results repo:", sync({json.dumps(files)}, {json.dumps(subject)}))
"""


def gpu_check(require_t4: bool) -> str:
    tail = ('assert "T4" in gpu, "Use a Standard T4: it is the reference GPU for the resume numbers."'
            if require_t4 else 'print("Any GPU is fine here; it is recorded with the results.")')
    return f'''
import torch
gpu = torch.cuda.get_device_name(0)
print(gpu, round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
{tail}
'''


BASE = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

NOTEBOOKS = {
    "01_baseline_eval": [
        ("md", """
# 01 — Baseline evaluation (Phase 6)

Zero-shot **Llama-3.2-3B-Instruct** (4-bit, no fine-tuning) on the untouched FinQA test set (1,147 questions), using the
same prompt format and scorer every later model (SFT, ORPO) will get.

**Runtime:** T4 or L4. Accuracy doesn't depend on the GPU, which is recorded with the results either way. About 2.5–3 hours on a T4
(the untrained model writes long answers).

**Progress is pushed to GitHub every 8 batches**, so a dropped connection or a recycled Colab machine costs at most a
few minutes. To resume, run the setup and install cells again, then the full-run cell. It skips everything already done.

**Token:** fine-grained GitHub token, *Contents: Read and write* on the private results repo only. Typed into a
hidden prompt, kept in memory, never printed or saved.
"""),
        ("code", SETUP),
        ("code", INSTALL),
        ("code", gpu_check(require_t4=False)),
        ("md", """
## Smoke test (16 questions)
Skip this if it already ran in an earlier session.
"""),
        ("code", f"!python -m src.eval.run_eval --model {BASE} --tag baseline --split test --limit 16 --batch-size 16"),
        ("code", '''
import json
preds = [json.loads(l) for l in open("results/baseline_test_limit16_predictions.jsonl")]
for p in preds[:2]:
    print(p["id"], "| correct:", p["correct"], "| extracted:", p["extracted"])
    print(p["output"][:1200])
    print("-" * 80)
'''),
        ("md", """
## Full test set (resumable)
If this fails with CUDA out-of-memory in the first batches, rerun with `--batch-size 8`.
"""),
        ("code", f"!python -m src.eval.run_eval --model {BASE} --tag baseline --split test --batch-size 16 --sync-every 8"),
        ("md", "## Final push (the run already pushes on completion; this is a safety net)"),
        ("code", push(["results", "experiments/runs"], "Baseline eval: Llama-3.2-3B-Instruct 4-bit on FinQA test")),
    ],
    "02_sft": [
        ("md", """
# 02 — QLoRA SFT (Phase 7) + SFT evaluation (Phase 8)

Fine-tunes Llama-3.2-3B-Instruct (4-bit NF4 base, LoRA on all attention + MLP projections) on the 4,508-example
curated FinQA set, then evaluates it on the FinQA test set with the same scorer as the baseline.

**Runtime: Standard T4.** This notebook produces the training numbers that go on the resume (peak VRAM, tokens/s),
and the T4 is the reference GPU for those.

**Tokens:** the GitHub token (as in notebook 01) and a Hugging Face **write** token. Checkpoints and the final adapter
go to a private HF repo every 100 steps, so a dropped session resumes with `--resume`.
"""),
        ("code", SETUP),
        ("code", '''
from huggingface_hub import login, whoami
login(token=getpass.getpass("Hugging Face write token: "), add_to_git_credential=False)
HF_USER = whoami()["name"]
print("HF user:", HF_USER)
'''),
        ("code", INSTALL),
        ("code", gpu_check(require_t4=True)),
        ("md", """
## 1. Probe: 20 steps, no eval (~5 min)
Catches setup errors cheaply and gives a first read on peak VRAM and throughput (batch 2 × grad-accum 8, rank 16).
"""),
        ("code", "!python -m src.train.sft --tag probe_r16 --rank 16 --max-steps 20 --no-eval"),
        ("md", """
## 2. Full SFT, rank 16 (2 epochs, ~564 steps)
If the session drops: rerun setup, HF login and install, then this cell with `--resume` added.
"""),
        ("code", '''
HUB_REPO = f"{HF_USER}/finqa-llama32-3b-sft-r16"
!python -m src.train.sft --tag sft_r16 --rank 16 --hub-repo {HUB_REPO}
'''),
        ("code", push(["results", "experiments/runs"], "SFT r16: training metrics")),
        ("md", "## 3. Evaluate the SFT model on FinQA test (resumable, pushes progress)"),
        ("code", "!python -m src.eval.run_eval --model outputs/sft_r16/adapter --tag sft_r16 --split test --batch-size 16 --sync-every 8"),
        ("code", '''
import json
base_path = "results/baseline_test_metrics.json"
base = json.load(open(base_path)) if os.path.exists(base_path) else None
sft = json.load(open("results/sft_r16_test_metrics.json"))
for k in ["accuracy", "accuracy_label_consistent", "answer_line_rate", "mean_output_tokens"]:
    print(f"{k:28s} base={base[k] if base else 'n/a'}  sft={sft[k]}")
'''),
        ("code", push(["results", "experiments/runs"], "SFT r16: FinQA test eval")),
        ("md", """
## Optional: LoRA rank sweep (r = 8, 32)
Only if compute units allow. Each is a full training job; compare eval loss, test accuracy, VRAM and tokens/s.
"""),
        ("code", '''
# for r in (8, 32):
#     !python -m src.train.sft --tag sft_r{r} --rank {r} --hub-repo {HF_USER}/finqa-llama32-3b-sft-r{r}
#     !python -m src.eval.run_eval --model outputs/sft_r{r}/adapter --tag sft_r{r} --split test --batch-size 16 --sync-every 8
'''),
    ],
    "04_align": [
        ("md", """
# 04 — Preference pairs (Phase 9) and ORPO (Phase 10)

One cell starts a background job (`src/jobs.py align`) that runs, in order, skipping anything already done:
1. fetch the SFT adapter from your HF repo (only on a fresh machine)
2. sample 4 answers per held-out ORPO prompt from the SFT model and score them (resumes from GitHub)
3. build the preference pairs (seconds)
4. ORPO training from the SFT adapter (~70 min; checkpoints to HF every 25 steps, resumes after a restart)
5. ORPO on the FinQA test set (~30 min)

TAT-QA evaluations run locally on the laptop with llama.cpp, not here.

**Colab Pro ends a session after ~90 minutes without user interaction**, even while code runs. While the job
runs, **click into this notebook and run the status cell at least once an hour**. If the machine is gone anyway,
start a new one and run this notebook again: finished work is skipped and the rest resumes.

**Runtime: Standard T4.** Tokens: GitHub and Hugging Face (write).
"""),
        ("code", SETUP),
        ("code", '''
from huggingface_hub import login, whoami
login(token=getpass.getpass("Hugging Face write token: "), add_to_git_credential=False)
print("HF user:", whoami()["name"])
'''),
        ("code", INSTALL),
        ("code", gpu_check(require_t4=True)),
        ("md", """
## Start the job and keep this cell running

The job runs in the background, but **this cell keeps running until the job finishes**, printing a status line every
5 minutes. Colab recycles machines whose notebook sits idle even while background work runs, so leave this cell
running and this tab open. If VS Code loses the connection, the job carries on; reopen the notebook and run the
status cell below.
"""),
        ("code", '''
import json, subprocess, time
if subprocess.run(["pgrep", "-f", "src.jobs align"], capture_output=True).returncode:
    subprocess.Popen("mkdir -p logs && nohup python -m src.jobs align > logs/job_align.log 2>&1 &", shell=True)
    time.sleep(30)
while subprocess.run(["pgrep", "-f", "src.jobs align"], capture_output=True).returncode == 0:
    try:
        d = json.load(open("results/job_align_progress.json"))
        print(time.strftime("%H:%M"), "step:", d["current"], "|", (d["log_tail"] or [""])[-1][:120], flush=True)
    except Exception as e:
        print(time.strftime("%H:%M"), "status unavailable:", e, flush=True)
    time.sleep(300)
print("job finished:", json.dumps(json.load(open("results/job_align_progress.json"))["steps"], indent=1))
'''),
        ("md", "## Check status any time"),
        ("code", '''
!cat results/job_align_progress.json
!nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
'''),
    ],
    "03_throughput_benchmark": [
        ("md", """
# 03 — Training throughput: Unsloth vs. Hugging Face + PEFT (Phase 7b)

Measures the claim behind "2.4× throughput speedup". The same QLoRA job (same NF4 checkpoint, LoRA rank 16 on all
projections, first 400 training examples in fixed order, batch 2 × grad-accum 8, fp16 compute forced on both, 8-bit AdamW,
gradient checkpointing) runs on two training stacks. 25 optimizer steps each; the first 3 are warm-up and excluded.

**Runtime: Standard T4**, the reference GPU. Needs only the GitHub token. About 15–25 minutes. Don't run anything else
on the same server at the same time: it would distort the timings.
"""),
        ("code", SETUP),
        ("code", INSTALL),
        ("code", gpu_check(require_t4=True)),
        ("md", "## 1. Hugging Face + PEFT + bitsandbytes (separate process)"),
        ("code", "!python -m src.train.bench_throughput --backend hf"),
        ("md", "## 2. Unsloth (separate process)"),
        ("code", "!python -m src.train.bench_throughput --backend unsloth"),
        ("md", "## 3. Compare and push"),
        ("code", "!python -m src.train.bench_throughput --compare"),
        ("code", push(["results/bench_hf.json", "results/bench_unsloth.json", "results/bench_compare.json",
                       "experiments/runs"], "Throughput benchmark: Unsloth vs HF+PEFT QLoRA on T4")),
    ],
}


def cell(kind: str, src: str) -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    body = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    ast.parse(body)  # fail loudly on any syntax error before writing
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = {"cells": [cell(k, s) for k, s in cells],
              "metadata": {"accelerator": "GPU",
                           "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 4}
        path = NB_DIR / f"{name}.ipynb"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {path} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
