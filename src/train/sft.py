"""Phase 7: QLoRA SFT of Llama-3.2-3B-Instruct on the curated FinQA set, with Unsloth (runs on Colab).

- 4-bit NF4 base (frozen) + LoRA adapters on all attention and MLP projections
- loss on the assistant response only: prompt tokens masked, pre-tokenized in src/train/data.py (verified on every
  train/val example), trained with transformers.Trainer + a padding collator
- Unsloth gradient checkpointing, 8-bit AdamW, fp16 on T4 (no bf16 support)
- measures what the resume claims are made of: peak VRAM (allocated and reserved), training tokens/s, eval loss

Checkpoints go to a private Hugging Face repo every --save-steps (hub_strategy="checkpoint"), so a dropped Colab
session resumes from the last pushed checkpoint with --resume.

Usage:
  python -m src.train.sft --tag sft_r16 --rank 16 --hub-repo <user>/finqa-llama32-3b-sft-r16
  python -m src.train.sft --tag bench --rank 16 --max-steps 30 --no-eval   # short throughput/VRAM probe
"""

import argparse
import json
import math
import time
from pathlib import Path

from src.train.data import build_dataset

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, help="LoRA alpha (default: = rank)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (throughput/VRAM probes)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--hub-repo", help="private HF repo for checkpoints + final adapter")
    ap.add_argument("--resume", action="store_true", help="resume from the last checkpoint pushed to --hub-repo")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    alpha = args.alpha or args.rank

    import unsloth  # noqa: F401  (must precede transformers/peft)
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=args.max_seq_length,
                                                         load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=alpha, lora_dropout=0, bias="none", target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4-bit weights are packed two per byte (Params4bit.numel() is half the logical count), as Unsloth reports.
    total = sum(p.numel() * (2 if type(p).__name__ == "Params4bit" else 1) for p in model.parameters())

    tokenizer.padding_side = "right"
    train_ds = build_dataset("sft_train", tokenizer, response_only=True)
    eval_ds = None if args.no_eval else build_dataset("sft_val", tokenizer, response_only=True)
    train_tokens_per_epoch = sum(len(ids) for ids in train_ds["input_ids"])

    out_dir = Path("outputs") / args.tag
    # With validation data, keep the checkpoint with the lowest eval loss rather than the last one: the loss on these
    # short, structured answers drops fast, so late epochs can overfit.
    keep_best = eval_ds is not None
    config = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        optim="adamw_8bit", fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=10, eval_strategy="no" if args.no_eval else "steps", eval_steps=args.eval_steps,
        save_strategy="steps" if (args.hub_repo or keep_best) else "no",
        save_steps=args.eval_steps if keep_best else args.save_steps, save_total_limit=2,
        load_best_model_at_end=keep_best, metric_for_best_model="eval_loss", greater_is_better=False,
        push_to_hub=bool(args.hub_repo), hub_model_id=args.hub_repo, hub_strategy="checkpoint",
        hub_private_repo=True, report_to="none", seed=args.seed)
    trainer = Trainer(model=model, args=config, train_dataset=train_ds, eval_dataset=eval_ds,
                      processing_class=tokenizer,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))

    resume = None
    if args.resume and args.hub_repo:
        from huggingface_hub import snapshot_download
        resume = snapshot_download(args.hub_repo, allow_patterns=["last-checkpoint/*"]) + "/last-checkpoint"

    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume)
    wall = time.perf_counter() - t0

    steps = result.global_step
    effective_batch = args.batch_size * args.grad_accum
    epochs_done = min(args.epochs, steps * effective_batch / len(train_ds))
    tokens_seen = train_tokens_per_epoch * epochs_done
    metrics = {
        "tag": args.tag, "base_model": BASE_MODEL, "gpu": torch.cuda.get_device_name(0),
        "lora": {"rank": args.rank, "alpha": alpha, "target_modules": TARGET_MODULES,
                 "trainable_params": trainable, "trainable_pct": round(100 * trainable / total, 3)},
        "train": {"examples": len(train_ds), "epochs": round(epochs_done, 3), "steps": steps,
                  "effective_batch": effective_batch, "lr": args.lr, "max_seq_length": args.max_seq_length,
                  "precision": "bf16" if is_bfloat16_supported() else "fp16", "resumed": bool(resume)},
        "final_train_loss": round(result.training_loss, 4),
        "train_runtime_s": round(result.metrics.get("train_runtime", wall), 1),
        "train_tokens_per_s": round(tokens_seen / result.metrics.get("train_runtime", wall), 1),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }
    if eval_ds is not None:
        ev = trainer.evaluate()  # the loaded best checkpoint
        metrics["final_eval_loss"] = round(ev["eval_loss"], 4)
        metrics["final_eval_ppl"] = round(math.exp(ev["eval_loss"]), 3)
        metrics["best_checkpoint"] = trainer.state.best_model_checkpoint
        metrics["best_eval_loss"] = trainer.state.best_metric
    metrics["log_history"] = [{k: v for k, v in h.items() if k in ("step", "loss", "eval_loss", "learning_rate")}
                              for h in trainer.state.log_history]

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    if args.hub_repo:
        model.push_to_hub(args.hub_repo, private=True)
        tokenizer.push_to_hub(args.hub_repo, private=True)

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.tag}_train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.max_steps < 0:
        from src.experiment_logger import log_run
        log_run(f"train_{args.tag}", {**metrics["lora"], **metrics["train"], "gpu": metrics["gpu"]},
                {k: v for k, v in metrics.items() if k not in ("lora", "train", "log_history", "tag", "gpu")})
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
