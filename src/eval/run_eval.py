"""Batched greedy evaluation of a chat model on a final split (runs on Colab).

Every model in the project (base, SFT, ORPO) is evaluated with this script, the same prompt format (system prompt +
report excerpt + question) and the same scorer, so accuracies are directly comparable.

Predictions are appended to a JSONL file as each batch finishes; rerunning the same command resumes where it stopped.

Usage (from the repo root):
  python -m src.eval.run_eval --model unsloth/Llama-3.2-3B-Instruct-bnb-4bit --tag baseline --split test
  python -m src.eval.run_eval ... --limit 16          # quick smoke test
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from src.colab_sync import sync
from src.data.render import CHAT_TEMPLATE_KWARGS
from src.eval.scoring import REL_TOL_LENIENT, score_record

RESULTS_DIR = Path("results")


def load_split(split: str, limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in Path(f"data/final/{split}.jsonl").open(encoding="utf-8")]
    return rows[:limit] if limit else rows


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def summarize(rows: list[dict], preds: dict[str, dict]) -> dict:
    scored = [(r, preds[r["id"]]) for r in rows if r["id"] in preds]
    acc = lambda items: round(sum(p["correct"] for _, p in items) / len(items), 4) if items else None
    rate = lambda f: round(sum(f(r, p) for r, p in scored) / len(scored), 4)
    finqa = all(r.get("dataset", "finqa") == "finqa" for r, _ in scored)
    metrics = {
        "n": len(scored),
        "accuracy": acc(scored),
        "answer_line_rate": rate(lambda r, p: "answer:" in p["output"].lower()),
        "hit_max_new_tokens_rate": rate(lambda r, p: p["hit_limit"]),
        "mean_output_tokens": round(sum(p["output_tokens"] for _, p in scored) / len(scored), 1),
    }
    if finqa:
        by_steps = defaultdict(list)
        for r, p in scored:
            by_steps[min(r["program"].count("("), 3)].append((r, p))
        metrics["accuracy_strict_sign"] = rate(lambda r, p: score_record(r, p["output"], strict_sign=True)[0])
        metrics["accuracy_lenient_0.5pct"] = rate(lambda r, p: score_record(r, p["output"],
                                                                            rel_tol=REL_TOL_LENIENT)[0])
        metrics["accuracy_by_program_steps"] = {("3+" if k == 3 else str(k)): acc(v) for k, v in sorted(by_steps.items())}
        metrics["accuracy_yes_no"] = acc([(r, p) for r, p in scored if r["exe_ans"] in ("yes", "no")])
    else:
        by_type = defaultdict(list)
        for r, p in scored:
            by_type[r["answer_type"]].append((r, p))
        metrics["accuracy_by_answer_type"] = {k: acc(v) for k, v in sorted(by_type.items())}
    if "label_consistent" in rows[0]:
        metrics["accuracy_label_consistent"] = acc([(r, p) for r, p in scored if r["label_consistent"]])
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path (base or merged/adapter dir)")
    ap.add_argument("--tag", required=True, help="run name, e.g. baseline / sft_r16 / orpo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sync-every", type=int, default=0,
                    help="push predictions to GitHub every N batches (Colab; needs GIT_PUSH_URL)")
    args = ap.parse_args()

    import unsloth  # noqa: F401  (must precede transformers)
    import torch
    from unsloth import FastLanguageModel

    rows = load_split(args.split, args.limit)
    suffix = f"_limit{args.limit}" if args.limit else ""
    pred_path = RESULTS_DIR / f"{args.tag}_{args.split}{suffix}_predictions.jsonl"
    metrics_path = RESULTS_DIR / f"{args.tag}_{args.split}{suffix}_metrics.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    preds = {}
    if pred_path.exists():
        for line in pred_path.open(encoding="utf-8"):
            p = json.loads(line)
            preds[p["id"]] = p
    todo = [r for r in rows if r["id"] not in preds]
    print(f"{len(rows)} examples, {len(preds)} already done, {len(todo)} to generate")

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_seq_length,
                                                         load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")})

    prompts = {r["id"]: tokenizer.apply_chat_template(r["messages"][:2], add_generation_prompt=True, tokenize=False,
                                                      **CHAT_TEMPLATE_KWARGS)
               for r in todo}
    # Longest first: an out-of-memory batch shows up in the first minute, not after an hour.
    todo.sort(key=lambda r: -len(prompts[r["id"]]))

    torch.cuda.reset_peak_memory_stats()
    gen_tokens, gen_time = 0, 0.0
    if True:  # predictions are appended per batch below (the file is never held open across batches)
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            enc = tokenizer([prompts[r["id"]] for r in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
            torch.cuda.synchronize(); gen_time += time.perf_counter() - t0
            new = gen[:, enc["input_ids"].shape[1]:]
            records = []
            for r, ids in zip(batch, new):
                ids = [t for t in ids.tolist() if t != tokenizer.pad_token_id]
                n_out = next((i + 1 for i, t in enumerate(ids) if t in stop_ids), len(ids))
                text = tokenizer.decode(ids[:n_out], skip_special_tokens=True)
                correct, extracted = score_record(r, text)
                gen_tokens += n_out
                p = {"id": r["id"], "output": text, "extracted": extracted, "correct": correct,
                     "output_tokens": n_out, "hit_limit": n_out >= args.max_new_tokens}
                preds[r["id"]] = p
                records.append(json.dumps(p, ensure_ascii=False) + "\n")
            with pred_path.open("a", encoding="utf-8") as out:  # append and close every batch
                out.writelines(records)
            done = start + len(batch)
            batch_no = start // args.batch_size + 1
            if args.sync_every and batch_no % args.sync_every == 0:
                sync([str(pred_path)], f"WIP {args.tag} {args.split}: {len(preds)}/{len(rows)} predictions")
            print(f"{done}/{len(todo)}  running acc {sum(p['correct'] for p in preds.values()) / len(preds):.3f}"
                  f"  {gen_tokens / max(gen_time, 1e-9):.0f} tok/s", flush=True)

    metrics = summarize(rows, preds)
    metrics["run"] = {
        "model": args.model, "tag": args.tag, "split": args.split, "quantization": "bnb-4bit (Unsloth)",
        "decoding": "greedy", "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "gpu": torch.cuda.get_device_name(0), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "batched_generation_tok_s_this_session": round(gen_tokens / gen_time, 1) if gen_time else None,
        "git_commit": git_commit(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if not args.limit:
        from src.experiment_logger import RUNS_DIR, log_run
        log_run(f"eval_{args.tag}", metrics["run"], {k: v for k, v in metrics.items() if k != "run"})
        if args.sync_every:
            sync([str(pred_path), str(metrics_path), str(RUNS_DIR)],
                 f"Eval {args.tag} on {args.split}: accuracy {metrics['accuracy']}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
