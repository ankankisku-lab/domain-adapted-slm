"""Evaluate a GGUF model with llama.cpp's llama-server on the local GPU (Phases 15-16, 18).

Same prompts as training and Colab evaluation (HF chat template, pinned date; the template's BOS is stripped because
llama-server adds one) and the same scorer, so accuracy is comparable across base/SFT/ORPO and every quant level.
Also records serving speed per request from llama-server's own timings: time to first token (prompt processing),
generation tokens/s, end-to-end latency, plus GPU memory used while the model is loaded.

Usage:
  python -m src.eval.run_eval_gguf --gguf models/orpo_r16/orpo_r16-Q4_K_M.gguf --tag orpo_r16_q4km --split test
  python -m src.eval.run_eval_gguf ... --limit 50 --parallel 1      # single-stream latency measurement
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import src.paths  # noqa: F401
from src.data.render import CHAT_TEMPLATE_KWARGS
from src.eval.run_eval import RESULTS_DIR, load_split, summarize
from src.eval.scoring import score_record

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "tools/llama.cpp/b11209/bin/llama-server.exe"
TOKENIZER = "unsloth/Llama-3.2-3B-Instruct"


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(url: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup (see the server log)")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("llama-server didn't become healthy")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--parallel", type=int, default=2, help="server slots / concurrent requests")
    ap.add_argument("--ctx-per-slot", type=int, default=2560)
    ap.add_argument("--gpu-layers", type=int, default=99, help="-ngl; 99 = everything that exists")
    ap.add_argument("--fit", action="store_true",
                    help="let llama-server choose the GPU layer count to fit VRAM (for models that don't fit fully)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rows = load_split(args.split, args.limit)
    prompts = {r["id"]: tok.apply_chat_template(r["messages"][:2], add_generation_prompt=True, tokenize=False,
                                                **CHAT_TEMPLATE_KWARGS).removeprefix(tok.bos_token) for r in rows}

    suffix = f"_limit{args.limit}" if args.limit else ""
    name = f"{args.tag}_{args.split}{suffix}"
    pred_path, metrics_path = RESULTS_DIR / f"{name}_predictions.jsonl", RESULTS_DIR / f"{name}_metrics.json"
    RESULTS_DIR.mkdir(exist_ok=True)

    vram_idle = gpu_used_mib()
    log = (ROOT / "logs" / f"llama_server_{name}.log").open("w", encoding="utf-8")
    placement = ["--fit", "on"] if args.fit else ["-ngl", str(args.gpu_layers)]
    proc = subprocess.Popen([str(SERVER), "-m", args.gguf, *placement,
                             "-c", str(args.ctx_per_slot * args.parallel), "-np", str(args.parallel),
                             "--port", str(args.port), "--no-webui", "-fa", "auto", "-lv", "4"], stdout=log, stderr=log)
    url = f"http://127.0.0.1:{args.port}"
    try:
        wait_healthy(url, proc)
        vram_loaded = gpu_used_mib()

        def one(r: dict) -> dict:
            t0 = time.perf_counter()
            out = post(url + "/completion", {"prompt": prompts[r["id"]], "n_predict": args.max_new_tokens,
                                              "temperature": 0.0, "top_k": 1, "stop": ["<|eot_id|>"],
                                              "cache_prompt": False})
            wall_ms = (time.perf_counter() - t0) * 1000
            text, t = out["content"], out.get("timings", {})
            correct, extracted = score_record(r, text)
            return {"id": r["id"], "output": text, "extracted": extracted, "correct": correct,
                    "output_tokens": t.get("predicted_n", 0),
                    "hit_limit": t.get("predicted_n", 0) >= args.max_new_tokens,
                    "prompt_tokens": t.get("prompt_n"), "ttft_ms": t.get("prompt_ms"),
                    "gen_tok_s": t.get("predicted_per_second"), "wall_ms": round(wall_ms, 1)}

        t_start = time.perf_counter()
        with ThreadPoolExecutor(args.parallel) as pool, pred_path.open("w", encoding="utf-8") as f:
            preds = {}
            for i, p in enumerate(pool.map(one, rows), 1):
                preds[p["id"]] = p
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                if i % 50 == 0:
                    acc = sum(x["correct"] for x in preds.values()) / len(preds)
                    print(f"{i}/{len(rows)} running acc {acc:.3f}", flush=True)
        elapsed = time.perf_counter() - t_start
        vram_peak = gpu_used_mib()
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        log.close()

    import re
    server_log = (ROOT / "logs" / f"llama_server_{name}.log").read_text(encoding="utf-8", errors="replace")
    offload = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", server_log)
    metrics = summarize(rows, preds)
    ps = list(preds.values())
    gen = [p["gen_tok_s"] for p in ps if p["gen_tok_s"]]
    metrics["serving"] = {
        "gguf": args.gguf, "gguf_size_gb": round(Path(args.gguf).stat().st_size / 1024**3, 3),
        "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)", "llama_cpp_build": "b11209",
        "parallel_slots": args.parallel, "placement": "fit" if args.fit else f"ngl {args.gpu_layers}",
        "layers_on_gpu": f"{offload[-1][0]}/{offload[-1][1]}" if offload else None,
        "vram_model_loaded_mib": vram_loaded - vram_idle, "vram_after_run_mib": vram_peak - vram_idle,
        "gen_tok_s_median": round(statistics.median(gen), 1) if gen else None,
        "gen_tok_s_p10": round(pct(gen, 0.10), 1) if gen else None,
        "ttft_ms_median": round(statistics.median(p["ttft_ms"] for p in ps), 1),
        "ttft_ms_p95": round(pct([p["ttft_ms"] for p in ps], 0.95), 1),
        "latency_ms_p50": round(pct([p["wall_ms"] for p in ps], 0.50), 1),
        "latency_ms_p95": round(pct([p["wall_ms"] for p in ps], 0.95), 1),
        "throughput_questions_per_min": round(60 * len(ps) / elapsed, 1),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.limit:
        from src.experiment_logger import log_run
        log_run(f"eval_gguf_{args.tag}", metrics["serving"], {k: v for k, v in metrics.items() if k != "serving"})
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
