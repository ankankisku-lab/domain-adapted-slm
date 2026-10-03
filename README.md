# Financial Reasoning SLM

A resource-constrained post-training pipeline that turns **Llama-3.2-3B-Instruct** into a local assistant for numerical questions over financial reports: verified domain data → 4-bit QLoRA fine-tuning on a single T4 → GGUF quantization → llama.cpp on a 4 GB laptop GPU. Open-weight models only, no paid APIs, and every number below comes from a logged run in this repository.

## Results

**In-domain: FinQA test set (1,147 questions, exact match at the stated precision)**

| Model | Accuracy | ±0.5% tolerance | Yes/no | Mean answer length |
|---|---:|---:|---:|---:|
| Llama-3.2-3B-Instruct, zero-shot (4-bit) | 28.1% | 41.5% | 55% | 161 tokens |
| + QLoRA SFT (this repo) | **42.5%** | 65.5% | 90% | 33.5 tokens |

The gain is +14.4 points (paired exact McNemar test: 273 vs 108 discordant answers, p < 0.0001). The gap between exact and ±0.5% accuracy shows the remaining weakness is arithmetic precision.

**Deployment: quantization study on an RTX 3050 Laptop GPU (4 GB), llama.cpp, same 1,147 questions**

| GGUF quant | Size | Accuracy | Layers on GPU | VRAM | Median latency |
|---|---:|---:|---:|---:|---:|
| f16 | 5.99 GB | 41.9% | 8 / 29 | GPU full, rest on CPU | 8.4 s |
| Q8_0 | 3.19 GB | 41.4% | 15 / 29 | GPU full, rest on CPU | 4.8 s |
| Q6_K | 2.46 GB | 41.3% | all | 3.2 GB | 1.8 s |
| **Q5_K_M** | **2.16 GB** | **41.8%** | **all** | **2.9 GB** | **1.6 s** |
| Q4_K_M + imatrix | 1.88 GB | 40.2% | all | 2.6 GB | 1.6 s |
| Q4_K_M | 1.88 GB | 39.1% | all | 2.6 GB | 1.6 s |

Q5_K_M matches the 16-bit model (McNemar p = 1.0) and fits entirely on the GPU. Q4_K_M loses a significant 2.8 points (p = 0.006); the importance-matrix variant's +1.1 points over plain Q4_K_M is not significant (p = 0.28). Single-stream generation with Q4_K_M is 69.6 tokens/s (`llama-bench`).

**Training efficiency: the same QLoRA job on a T4 (fp16, identical settings)**

| Stack | Throughput | Peak VRAM (reserved) |
|---|---:|---:|
| Hugging Face + PEFT | 475.8 tokens/s | 13.74 GB |
| Unsloth | 698.5 tokens/s (1.47×) | 4.33 GB (−68.5%) |

**External check: TAT-QA (500-question stratified subset, never trained on)**

| Answer type | Base | SFT |
|---|---:|---:|
| Overall | 52.6% | 32.8% |
| Arithmetic | 55.2% | 54.3% |
| Text span | 50.7% | 15.3% |
| Multi-span | 52.4% | 25.4% |
| Count | 41.7% | 8.3% |

This is a known limitation. FinQA contains only numeric answers, so the fine-tuned model forces a calculation onto every question (format overfitting). Arithmetic transfers; lookup and text answers regress. The planned fix is to mix non-numeric examples into fine-tuning.

## Status

| Stage | State |
|---|---|
| Data pipeline (cleaning, label verification, deduplication, splits) | Done |
| Baseline evaluation | Done |
| QLoRA SFT and evaluation | Done |
| Unsloth vs HF + PEFT benchmark | Done |
| GGUF conversion, quantization study, llama.cpp evaluation | Done |
| TAT-QA external evaluation | Done (base and SFT) |
| Preference pairs for ORPO | In progress (824 of 1,212 prompts sampled) |
| ORPO alignment | Implemented and unit-tested; **not trained yet** |
| FastAPI service | Implemented; not demoed yet |

## How it works

**Data (FinQA, 8,281 examples → 6,025 verified → 4,508 for SFT).**

- Regex cleaning of PDF-extraction artifacts, checked by an invariant that no number may be lost from a context.
- Label verification: an executor for FinQA's reasoning-program language reproduces the recorded result for 8,279 of 8,281 programs; 639 training examples whose annotated answer disagrees with the executed program are dropped.
- Deduplication with BGE-small embeddings. The duplicate rule (same executed answer, same program inputs, similarity ≥ 0.86) was chosen on 200 labelled pairs: precision 0.90, recall 0.98, versus F1 0.09 for similarity alone. The pairs were labelled with LLM assistance; see `data/labels/dedup_labels_provenance.json`.
- Splits grouped by company-year filing, so no 10-K contributes pages to two splits: 4,508 SFT train, 305 validation, 1,212 held-out prompts for preference data. The official test set is untouched.

**Training.** 4-bit NF4 base with LoRA (rank 16, all attention and MLP projections; 24.3M trainable parameters, 0.75%), response-only loss, 8-bit AdamW, 2 epochs on a T4, best checkpoint by validation loss. Peak VRAM is 4.25 GB during training steps.

**Evaluation.** One prompt format and one deterministic scorer for every model (`src/eval/scoring.py`), no LLM judge. Comparisons use paired exact McNemar tests.

**Deployment.** LoRA merged into the 16-bit base, converted to GGUF, quantized with `llama-quantize`, served with `llama-server`; `src/serve/api.py` adds a small FastAPI layer.

## Repository layout

| Path | Contents |
|---|---|
| `src/data/` | Download, schema, cleaning, program executor, deduplication, quality filters, splits |
| `src/train/` | QLoRA SFT, ORPO trainer, throughput benchmark, pre-tokenization |
| `src/pref/` | Sampling from the SFT model and the preference-pair rubric |
| `src/eval/` | Shared scorer, Colab evaluation, llama.cpp evaluation, offline rescoring |
| `src/deploy/`, `src/serve/` | GGUF build and quantization; FastAPI service |
| `src/jobs.py`, `src/colab_sync.py` | Unattended Colab job runner and progress sync |
| `notebooks/` | Colab notebooks (generated by `scripts/build_notebooks.py`) |
| `data/final/` | SFT, validation, preference-prompt and test splits in chat format |
| `data/manifest/`, `reports/` | Data reports for each phase; HTML dataset report |
| `results/`, `experiments/runs/` | Metrics, predictions and run records |
| `tests/` | CPU tests for the ORPO objective |

## Reproducing

Data pipeline (CPU, a few minutes plus ~5 minutes of embedding):

```
pip install -r requirements-local.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m src.data.download
python -m src.data.standardize
python -m src.data.clean
python -m src.data.dedup embed
python -m src.data.dedup apply
python -m src.data.quality
python -m src.data.build_splits
```

Training and Colab evaluation (T4): run the notebooks in `notebooks/` in order. They install `requirements-colab.txt`, which pins the versions used for the reported runs.

Local deployment (needs a llama.cpp build under `tools/`):

```
python -m src.deploy.build_gguf --tag sft_r16 --adapter <adapter repo or dir> --quants Q5_K_M
python -m src.eval.run_eval_gguf --gguf models/sft_r16/sft_r16-Q5_K_M.gguf --tag sft_r16_q5km --split test
```

## Licences

The code is released under the MIT licence (`LICENSE`). Datasets and models keep their own licences; see `LICENSES.md`. Fine-tuned weights are derivatives of Llama 3.2 and are covered by the Llama 3.2 Community License.
