"""Phase 7b: QLoRA training throughput and peak VRAM, Unsloth vs. standard Hugging Face + PEFT (runs on Colab).

Both backends train the same prequantized NF4 checkpoint with the same LoRA config (rank 16, all attention + MLP
projections), data (the first N sft_train examples, pre-tokenized once, fixed order), trainer (transformers.Trainer
with a padding collator), batch (2 x grad-accum 8), precision (fp16 compute), optimizer (8-bit AdamW) and gradient
checkpointing. The only difference is the model stack:
  hf       transformers + peft + bitsandbytes, prepare_model_for_kbit_training, HF gradient checkpointing, SDPA
  unsloth  Unsloth's patched kernels and "unsloth" gradient checkpointing
Loss is on all tokens for both (identical compute; this measures speed, not quality).

Run each backend in its own process: importing Unsloth patches transformers globally.
  python -m src.train.bench_throughput --backend hf
  python -m src.train.bench_throughput --backend unsloth
  python -m src.train.bench_throughput --compare
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from src.train.data import build_dataset

MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANK, BATCH, ACCUM, MAX_LEN = 16, 2, 8, 2048
STEPS, WARMUP_STEPS = 25, 3
SEED = 3407


def load(backend: str):
    import torch
    if backend == "unsloth":
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(MODEL, max_seq_length=MAX_LEN, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(model, r=RANK, lora_alpha=RANK, lora_dropout=0, bias="none",
                                                 target_modules=TARGET_MODULES,
                                                 use_gradient_checkpointing="unsloth", random_state=SEED)
        return model, tok, False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = "<|finetune_right_pad_id|>"  # the pad token Unsloth uses for Llama 3.x, for identical batches
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    # The checkpoint's quantization config says bnb_4bit_compute_dtype=bfloat16. The T4 has no native bf16, so
    # leaving it would slow the HF baseline artificially (Unsloth switches to fp16 on T4 itself). Force fp16.
    import bitsandbytes as bnb
    for m in model.modules():
        if isinstance(m, bnb.nn.Linear4bit):
            m.compute_dtype = torch.float16
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=RANK, lora_dropout=0.0, bias="none",
                                             target_modules=TARGET_MODULES, task_type="CAUSAL_LM"))
    return model, tok, True


def run(backend: str) -> None:
    if backend == "unsloth":
        import unsloth  # noqa: F401  (must be imported before transformers/peft or its patches don't apply)
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

    torch.manual_seed(SEED)
    model, tok, hf_checkpointing = load(backend)
    tok.padding_side = "right"  # training padding, identical for both backends
    n_examples = BATCH * ACCUM * STEPS
    dataset = build_dataset("sft_train", tok, response_only=False, limit=n_examples)
    lengths = [len(ids) for ids in dataset["input_ids"]]
    tokens_per_step = [sum(lengths[i * BATCH * ACCUM:(i + 1) * BATCH * ACCUM]) for i in range(STEPS)]

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.times, self._t = [], None

        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize(); self._t = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize(); self.times.append(time.perf_counter() - self._t)

    timer = StepTimer()
    cfg = TrainingArguments(output_dir=f"outputs/bench_{backend}", per_device_train_batch_size=BATCH,
                            gradient_accumulation_steps=ACCUM, max_steps=STEPS, learning_rate=2e-4,
                            lr_scheduler_type="constant", optim="adamw_8bit", fp16=True, bf16=False,
                            gradient_checkpointing=hf_checkpointing,
                            gradient_checkpointing_kwargs={"use_reentrant": False} if hf_checkpointing else None,
                            logging_steps=5, save_strategy="no", report_to="none", seed=SEED, data_seed=SEED)
    trainer = Trainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok, callbacks=[timer],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    # Fixed order, no shuffling: both backends see the same batches, so per-step token counts line up.
    trainer._get_train_sampler = lambda *a, **k: torch.utils.data.SequentialSampler(trainer.train_dataset)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()

    measured = list(zip(timer.times, tokens_per_step))[WARMUP_STEPS:]
    step_s = [t for t, _ in measured]
    import bitsandbytes as bnb
    compute_dtypes = sorted({str(m.compute_dtype) for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)})
    result = {
        "backend": backend, "linear4bit_compute_dtypes": compute_dtypes, "gpu": torch.cuda.get_device_name(0), "model": MODEL,
        "config": {"rank": RANK, "batch": BATCH, "grad_accum": ACCUM, "max_len": max(lengths), "steps": STEPS,
                   "warmup_steps_excluded": WARMUP_STEPS, "precision": "fp16", "optimizer": "adamw_8bit"},
        "median_step_s": round(statistics.median(step_s), 3),
        "tokens_per_s": round(sum(n for _, n in measured) / sum(step_s), 1),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "final_logged_loss": next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench_{backend}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def compare() -> None:
    hf, us = (json.loads(Path(f"results/bench_{b}.json").read_text()) for b in ("hf", "unsloth"))
    assert hf["gpu"] == us["gpu"], "backends were benchmarked on different GPUs"
    assert hf["linear4bit_compute_dtypes"] == us["linear4bit_compute_dtypes"], "backends used different compute dtypes"
    out = {"gpu": us["gpu"],
           "throughput_speedup": round(us["tokens_per_s"] / hf["tokens_per_s"], 2),
           "tokens_per_s": {"hf": hf["tokens_per_s"], "unsloth": us["tokens_per_s"]},
           "peak_vram_reserved_gb": {"hf": hf["peak_vram_reserved_gb"], "unsloth": us["peak_vram_reserved_gb"]},
           "peak_vram_allocated_gb": {"hf": hf["peak_vram_allocated_gb"], "unsloth": us["peak_vram_allocated_gb"]},
           "vram_reduction_pct": round(100 * (1 - us["peak_vram_reserved_gb"] / hf["peak_vram_reserved_gb"]), 1)}
    Path("results/bench_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    from src.experiment_logger import log_run
    log_run("bench_unsloth_vs_hf", {**us["config"], "gpu": us["gpu"]}, out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "unsloth"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    compare() if a.compare else run(a.backend)
