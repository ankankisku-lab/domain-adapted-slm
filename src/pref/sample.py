"""Phase 9a: sample several answers per prompt from the SFT model and score each (runs on Colab).

For the 1,212 ORPO prompts (held out of SFT training), draws --k samples per prompt at --temperature and scores
every sample against the executed gold answer with the shared scorer. Prompts where the model is sometimes right and
sometimes wrong give on-policy chosen/rejected pairs (src/pref/build_pairs.py).

Resumable (skips prompts already in the output file) and pushes progress with --sync-every, like run_eval.

Usage:
  python -m src.pref.sample --model outputs/sft_r16/adapter --tag sft_r16 --k 4 --temperature 0.8 --sync-every 10
"""

import argparse
import json
import time
import zlib
from pathlib import Path

from src.colab_sync import sync
from src.data.render import CHAT_TEMPLATE_KWARGS
from src.eval.run_eval import load_split
from src.eval.scoring import score_record

RESULTS_DIR = Path("results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="pref_prompts")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prompts-per-batch", type=int, default=4, help="batch = prompts-per-batch x k sequences")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=3072)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sync-every", type=int, default=0)
    args = ap.parse_args()

    import unsloth  # noqa: F401  (must precede transformers)
    import torch
    from unsloth import FastLanguageModel

    rows = load_split(args.split, args.limit)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = RESULTS_DIR / f"{args.tag}_{args.split}{suffix}_samples.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open(encoding="utf-8")}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} prompts, {len(done)} already sampled, {len(todo)} to go", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_seq_length,
                                                         load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")})

    prompts = {r["id"]: tokenizer.apply_chat_template(r["messages"][:2], add_generation_prompt=True, tokenize=False,
                                                      **CHAT_TEMPLATE_KWARGS) for r in todo}
    todo.sort(key=lambda r: -len(prompts[r["id"]]))  # longest first: OOM shows up early

    t_start, n_prompts, n_correct, n_mixed = time.perf_counter(), 0, 0, 0
    if True:  # results are appended per batch below (the file is never held open across batches)
        for b, start in enumerate(range(0, len(todo), args.prompts_per_batch), 1):
            batch = todo[start : start + args.prompts_per_batch]
            # Seed from the prompt ids, so a resumed run draws the same samples for the same prompts.
            torch.manual_seed(args.seed + sum(zlib.crc32(r["id"].encode()) for r in batch) % 2**31)
            enc = tokenizer([prompts[r["id"]] for r in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                                     temperature=args.temperature, top_p=args.top_p,
                                     num_return_sequences=args.k, pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=stop_ids)
            new = gen[:, enc["input_ids"].shape[1]:]
            records = []
            for i, r in enumerate(batch):
                samples = []
                for ids in new[i * args.k:(i + 1) * args.k]:
                    ids = [t for t in ids.tolist() if t != tokenizer.pad_token_id]
                    n_out = next((j + 1 for j, t in enumerate(ids) if t in stop_ids), len(ids))
                    text = tokenizer.decode(ids[:n_out], skip_special_tokens=True)
                    correct, extracted = score_record(r, text)
                    samples.append({"output": text, "extracted": extracted, "correct": correct,
                                    "output_tokens": n_out, "hit_limit": n_out >= args.max_new_tokens})
                n_prompts += 1
                n_correct += sum(s["correct"] for s in samples)
                n_mixed += 0 < sum(s["correct"] for s in samples) < args.k
                records.append(json.dumps({"id": r["id"], "k": args.k, "temperature": args.temperature,
                                           "samples": samples}, ensure_ascii=False) + "\n")
            # Append and close every batch: never hold the file open while anything else may replace it.
            with out_path.open("a", encoding="utf-8") as out:
                out.writelines(records)
            elapsed = time.perf_counter() - t_start
            print(f"{start + len(batch)}/{len(todo)} prompts  sample acc {n_correct / (n_prompts * args.k):.3f}"
                  f"  mixed {n_mixed / n_prompts:.2f}  {elapsed / n_prompts:.1f}s/prompt", flush=True)
            if args.sync_every and b % args.sync_every == 0:
                sync([str(out_path)], f"WIP samples {args.tag}: {len(done) + n_prompts}/{len(rows)} prompts")

    if args.sync_every:
        sync([str(out_path)], f"Samples {args.tag} on {args.split}: {len(done) + n_prompts} prompts x k={args.k}")


if __name__ == "__main__":
    main()
