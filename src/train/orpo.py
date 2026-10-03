"""Phase 10: ORPO preference alignment of the SFT model (runs on Colab), reference-model-free.

Objective (Hong et al., 2024, "ORPO: Monolithic Preference Optimization without Reference Model"):
  L = L_NLL(chosen) + beta * L_OR,   L_OR = -log sigmoid( log odds(chosen) - log odds(rejected) )
  log odds(y|x) = log p - log(1 - p), with log p the length-normalised (mean per token) log-likelihood of the response.
No frozen reference model is kept in memory: the only model is the policy being trained.

Implemented here on transformers.Trainer instead of TRL's ORPOTrainer, whose dataset.map-based tokenization crashes
under Unsloth on Colab's Python 3.13 (same issue as SFT, see src/train/data.py). The loss is unit-tested against a
direct computation on a tiny model (tests/test_orpo_loss.py).

Starts from the SFT LoRA adapter: same LoRA config, weights loaded from the SFT adapter, training continues.

Usage:
  python -m src.train.orpo --tag orpo_r16 --sft-adapter outputs/sft_r16/adapter --hub-repo <user>/finqa-llama32-3b-orpo
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from src.train.data import IGNORE, tokenize_example

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PAIRS_PATH = Path("data/final/pref_pairs.jsonl")


# ---------- data ----------

def load_pairs(val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split pairs by source prompt, so both pairs from one prompt land on the same side."""
    pairs = [json.loads(line) for line in PAIRS_PATH.open(encoding="utf-8")]
    sources = sorted({p["source_id"] for p in pairs})
    random.Random(seed).shuffle(sources)
    val_sources = set(sources[:max(1, int(len(sources) * val_fraction))])
    return ([p for p in pairs if p["source_id"] not in val_sources],
            [p for p in pairs if p["source_id"] in val_sources])


def tokenize_pair(pair: dict, tokenizer) -> dict:
    from src.data.render import CHAT_TEMPLATE_KWARGS

    def encode(response: list[dict]) -> dict:
        text = (tokenizer.apply_chat_template(pair["prompt"] + response, tokenize=False, **CHAT_TEMPLATE_KWARGS)
                .removeprefix(tokenizer.bos_token))
        return tokenize_example(text, tokenizer, response_only=True)

    c, r = encode(pair["chosen"]), encode(pair["rejected"])
    return {"chosen_input_ids": c["input_ids"], "chosen_labels": c["labels"],
            "rejected_input_ids": r["input_ids"], "rejected_labels": r["labels"]}


class PairCollator:
    """Pads chosen and rejected sequences to one length and stacks them: rows [0, B) chosen, [B, 2B) rejected."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict:
        import torch
        seqs = [(f["chosen_input_ids"], f["chosen_labels"]) for f in features] + \
               [(f["rejected_input_ids"], f["rejected_labels"]) for f in features]
        width = max(len(ids) for ids, _ in seqs)
        input_ids = [ids + [self.pad_id] * (width - len(ids)) for ids, _ in seqs]
        labels = [lab + [IGNORE] * (width - len(lab)) for _, lab in seqs]
        mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in seqs]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask)}


# ---------- loss ----------

def response_logps(logits, labels):
    """Mean log-prob per response token (labels == IGNORE are excluded), plus the per-row token count.

    The log-softmax is taken only at response positions (~40 tokens), not over the whole ~1,000-token sequence x
    128K vocabulary, which would cost several GB of activations per step."""
    import torch
    logits, labels = logits[:, :-1, :], labels[:, 1:]
    mask = labels != IGNORE
    rows = mask.nonzero(as_tuple=True)[0]
    token_logps = torch.gather(logits[mask].float().log_softmax(-1), 1, labels[mask].unsqueeze(-1)).squeeze(-1)
    counts = mask.sum(-1).clamp(min=1)
    sums = torch.zeros(labels.shape[0], device=logits.device, dtype=token_logps.dtype).index_add(0, rows, token_logps)
    return sums / counts, counts


def orpo_loss(logits, labels, beta: float) -> tuple:
    """Returns (loss, stats). Rows [0, B) are chosen, [B, 2B) rejected."""
    import torch
    import torch.nn.functional as F
    logps, counts = response_logps(logits, labels)
    b = logps.shape[0] // 2
    chosen, rejected = logps[:b], logps[b:]
    nll = -chosen.mean()  # mean over pairs of the per-token NLL of the chosen response
    # log odds = log p - log(1 - p); expm1 form keeps 1 - p stable when p is close to 1.
    log_odds = (chosen - rejected) - (torch.log(-torch.expm1(chosen)) - torch.log(-torch.expm1(rejected)))
    ratio = F.logsigmoid(log_odds)
    loss = nll - beta * ratio.mean()
    stats = {"nll": nll.detach(), "or_loss": -ratio.mean().detach(), "log_odds": log_odds.mean().detach(),
             "reward_acc": (chosen > rejected).float().mean().detach(),
             "margin": (chosen - rejected).mean().detach()}
    return loss, stats


def make_trainer_class():
    from transformers import Trainer

    class ORPOTrainer(Trainer):
        beta: float = 0.1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Our loss is a per-batch mean, so let the Trainer divide by the accumulation steps. (For Llama it would
            # otherwise assume the model normalised by num_items_in_batch and skip that, scaling gradients 8x.)
            self.model_accepts_loss_kwargs = False
            self._train_stats, self._eval_stats = [], None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs, use_cache=False)
            loss, stats = orpo_loss(outputs.logits, labels, self.beta)
            record = {k: float(v) for k, v in stats.items()}
            if model.training:
                self._train_stats.append(record)
            elif self._eval_stats is not None:
                self._eval_stats.append(record)
            return (loss, outputs) if return_outputs else loss

        @staticmethod
        def _mean(records: list[dict], prefix: str = "") -> dict:
            return {prefix + k: sum(r[k] for r in records) / len(records) for k in records[0]} if records else {}

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and self._train_stats:  # average over the logging interval, not the last batch
                logs.update(self._mean(self._train_stats))
                self._train_stats = []
            super().log(logs, *args, **kwargs)

        def evaluate(self, *args, **kwargs):
            self._eval_stats = []
            metrics = super().evaluate(*args, **kwargs)
            extra = self._mean(self._eval_stats, "eval_")
            self._eval_stats = None
            metrics.update(extra)
            self.log(extra)
            return metrics

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            import torch
            with torch.no_grad():
                loss = self.compute_loss(model, dict(inputs))
            return loss.detach(), None, None

    return ORPOTrainer


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sft-adapter", required=True, help="dir with the SFT LoRA adapter to continue from")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.1, help="weight of the odds-ratio term")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=1, help="pairs per device step (2 sequences each)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--hub-repo")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the last checkpoint pushed to --hub-repo, if there is one")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    # Unsloth skips materialising logits during training unless asked; the ORPO loss needs them.
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import unsloth  # noqa: F401  (must precede transformers/peft)
    import torch
    from datasets import Dataset
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainingArguments
    from unsloth import FastLanguageModel, is_bfloat16_supported

    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=args.max_seq_length,
                                                         load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth", random_state=args.seed)
    result = set_peft_model_state_dict(model, load_file(str(Path(args.sft_adapter) / "adapter_model.safetensors")))
    missing = [k for k in result.missing_keys if "lora_" in k]
    assert not missing and not result.unexpected_keys, f"SFT adapter didn't load cleanly: {missing[:3]}"

    train_pairs, val_pairs = load_pairs(args.val_fraction, args.seed)
    train_ds = Dataset.from_list([tokenize_pair(p, tokenizer) for p in train_pairs])
    val_ds = Dataset.from_list([tokenize_pair(p, tokenizer) for p in val_pairs])

    Trainer = make_trainer_class()
    Trainer.beta = args.beta
    out_dir = Path("outputs") / args.tag
    config = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.1, weight_decay=0.0, optim="adamw_8bit",
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=5,
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, remove_unused_columns=False, push_to_hub=bool(args.hub_repo),
        hub_model_id=args.hub_repo, hub_strategy="checkpoint", hub_private_repo=True, report_to="none",
        seed=args.seed)
    trainer = Trainer(model=model, args=config, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, data_collator=PairCollator(tokenizer.pad_token_id))

    resume = None
    if args.resume and args.hub_repo:
        from huggingface_hub import list_repo_files, snapshot_download
        try:
            if any(f.startswith("last-checkpoint/") for f in list_repo_files(args.hub_repo)):
                resume = snapshot_download(args.hub_repo, allow_patterns=["last-checkpoint/*"]) + "/last-checkpoint"
        except Exception as e:  # no repo yet: fresh start
            print(f"no checkpoint to resume from ({e.__class__.__name__}); starting fresh", flush=True)
    before = trainer.evaluate()
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume)
    wall = time.perf_counter() - t0
    after = trainer.evaluate()

    keys = ("eval_loss", "eval_nll", "eval_or_loss", "eval_reward_acc", "eval_margin", "eval_log_odds")
    metrics = {
        "tag": args.tag, "gpu": torch.cuda.get_device_name(0), "sft_adapter": args.sft_adapter,
        "config": {"beta": args.beta, "lr": args.lr, "epochs": args.epochs, "rank": args.rank,
                   "pairs_per_step": args.batch_size * args.grad_accum, "train_pairs": len(train_pairs),
                   "val_pairs": len(val_pairs), "reference_model": "none (ORPO)"},
        "val_before": {k: round(before[k], 4) for k in keys if k in before},
        "val_after": {k: round(after[k], 4) for k in keys if k in after},
        "best_checkpoint": trainer.state.best_model_checkpoint, "steps": result.global_step,
        "resumed_from_checkpoint": bool(resume),
        "train_runtime_s": round(result.metrics.get("train_runtime", wall), 1),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "log_history": trainer.state.log_history,
    }
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
        log_run(f"train_{args.tag}", {**metrics["config"], "gpu": metrics["gpu"]},
                {k: v for k, v in metrics.items() if k not in ("config", "log_history", "tag", "gpu")})
    print(json.dumps({k: v for k, v in metrics.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
