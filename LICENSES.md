# Licences

## This repository's code

MIT licence; see `LICENSE`.

## Datasets

The files under `data/` and the prediction files under `results/` contain text derived from the datasets below. They are redistributed under the datasets' own licences, with attribution.

| Dataset | Version used | Licence | Use here | What this repo redistributes |
| --- | --- | --- | --- | --- |
| FinQA (Chen et al., EMNLP 2021), "FinQA: A Dataset of Numerical Reasoning over Financial Data" | GitHub `czyssrs/FinQA@0f16e28` | MIT (GitHub repository); CC BY 4.0 (Hugging Face mirror `ibm-research/finqa`) | Fine-tuning, preference prompts, in-domain test | Cleaned and reformatted splits in `data/final/`, labelled duplicate pairs in `data/labels/`, model predictions |
| TAT-QA (Zhu et al., ACL 2021), "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance" | Hugging Face `next-tat/TAT-QA@c96247f` | CC BY 4.0 | External evaluation only; never trained on | A 500-question stratified subset of the test split, reformatted as prompts (`data/final/tatqa_test.jsonl`), and model predictions |

Changes made to the data: text normalisation, removal of examples with inconsistent labels or near-duplicates, rendering of tables as markdown, and rendering of FinQA reasoning programs as step-by-step answers.

## Models

| Model | Licence | Use here |
| --- | --- | --- |
| Llama-3.2-3B-Instruct (Meta; loaded from the `unsloth/` mirrors) | Llama 3.2 Community License | Base model. Fine-tuned adapters and GGUF files are derivatives ("Built with Llama") and carry the same licence. No model weights are stored in this repository. |
| BGE-small-en-v1.5 (BAAI) | MIT | Embeddings for deduplication |

## Tools

llama.cpp (MIT), Unsloth (Apache-2.0), Hugging Face Transformers, PEFT and Datasets (Apache-2.0), bitsandbytes (MIT).
