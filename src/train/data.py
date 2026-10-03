"""Pre-tokenized training data with response-only labels, built in a plain loop.

Why not TRL's own tokenization: Unsloth patches `datasets.Dataset.map` to run in worker processes, which pickles the
tokenization closure, and that fails on Colab's Python 3.13 ("cannot pickle 'ConfigModuleInstance'"). Tokenizing
here avoids `map` entirely and makes the loss mask explicit and testable.

Every example: input_ids (one BOS, from the tokenizer), attention_mask, and labels = -100 for the system + user turns
and the assistant header, token ids for the assistant response and its <|eot_id|>.
"""

import json
from pathlib import Path

from src.data.render import CHAT_TEMPLATE_KWARGS

RESPONSE_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"
IGNORE = -100


def chat_text(record: dict, tokenizer) -> str:
    """Full chat-formatted example without the template's BOS (the tokenizer adds exactly one)."""
    return (tokenizer.apply_chat_template(record["messages"], tokenize=False, **CHAT_TEMPLATE_KWARGS)
            .removeprefix(tokenizer.bos_token))


def _find_last(seq: list[int], sub: list[int]) -> int:
    for i in range(len(seq) - len(sub), -1, -1):
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


def tokenize_example(text: str, tokenizer, response_only: bool = True) -> dict:
    ids = tokenizer(text)["input_ids"]
    labels = list(ids)
    if response_only:
        header = tokenizer(RESPONSE_HEADER, add_special_tokens=False)["input_ids"]
        start = _find_last(ids, header)
        if start < 0:
            raise ValueError("assistant header not found in tokenized example")
        cut = start + len(header)
        labels = [IGNORE] * cut + ids[cut:]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load_records(split: str, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(f"data/final/{split}.jsonl").open(encoding="utf-8")]
    return rows[:limit] if limit else rows


def build_dataset(split: str, tokenizer, response_only: bool = True, limit: int | None = None):
    from datasets import Dataset

    examples = [tokenize_example(chat_text(r, tokenizer), tokenizer, response_only)
                for r in load_records(split, limit)]
    return Dataset.from_list(examples)
