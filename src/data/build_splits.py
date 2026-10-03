"""Phase 5a: final splits in chat format.

The verified pool (train + dev after Phases 2-4) is split by company-year filing, so no 10-K contributes pages to
two splits:
  sft_val       ~300 examples, SFT validation loss / early stopping
  pref_prompts  1,200 prompts reserved for ORPO. They're kept out of SFT training on purpose: on memorized prompts
                the SFT model would rarely produce wrong answers, leaving few "rejected" candidates
  sft_train     the rest
The official FinQA test set is rendered the same way but never filtered; it carries a label_consistent flag from
Phase 2 so results can be reported on all 1,147 and on the label-consistent subset.

Usage:  python -m src.data.build_splits
"""

import json
import random
from collections import Counter
from pathlib import Path

import src.paths  # noqa: F401  (keeps the HF cache on D:)
from src.data.render import SYSTEM_PROMPT, final_answer, render_prompt, render_response

POOL_PATH = Path("data/interim/finqa_quality.jsonl")
CLEAN_PATH = Path("data/interim/finqa_clean.jsonl")
RAW_PATH = Path("data/standard/finqa.jsonl")
OUT_DIR = Path("data/final")
REPORT_PATH = Path("data/manifest/splits_report.json")
TOKENIZER_REPO = "unsloth/Llama-3.2-3B-Instruct"

VAL_SIZE = 300
PREF_SIZE = 1200
MAX_SEQ_LEN = 2048
CHAT_TEMPLATE_OVERHEAD = 40  # Llama 3.2 header/eot tokens for system + user + assistant turns (upper bound)
SEED = 20260927


def to_example(rec: dict, response: str, split: str) -> dict:
    prompt = render_prompt(rec)
    return {
        "id": rec["id"], "split": split, "report_id": rec["report_id"], "company": rec["company"],
        "year": rec["year"], "question": rec["question"], "program": rec["program"],
        "exe_ans": rec["exe_ans"], "answer": final_answer(rec),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": prompt},
                     {"role": "assistant", "content": response}],
    }


def group_split(pool: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for rec in pool:
        groups.setdefault(f"{rec['company']}/{rec['year']}", []).append(rec)
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    splits: dict[str, list[dict]] = {"sft_val": [], "pref_prompts": [], "sft_train": []}
    for k in keys:
        if len(splits["sft_val"]) < VAL_SIZE:
            target = "sft_val"
        elif len(splits["pref_prompts"]) < PREF_SIZE:
            target = "pref_prompts"
        else:
            target = "sft_train"
        splits[target].extend(groups[k])
    return splits


def main() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    raw_tables = {}
    for line in RAW_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        raw_tables[r["id"]] = r["table"]

    def seq_tokens(ex: dict) -> int:
        return CHAT_TEMPLATE_OVERHEAD + sum(len(tokenizer.encode(m["content"]).ids) for m in ex["messages"])

    pool = [json.loads(line) for line in POOL_PATH.open(encoding="utf-8")]
    test = [json.loads(line) for line in CLEAN_PATH.open(encoding="utf-8")]
    test = [r for r in test if r["original_split"] == "test"]

    out: dict[str, list[dict]] = {}
    for name, recs in group_split(pool).items():
        out[name] = [to_example(r, r["response"], name) for r in recs]
    out["test"] = []
    for r in test:
        ex = to_example(r, render_response(r, raw_tables[r["id"]]) or "", "test")
        ex["label_consistent"] = r["check_info"]["answer_agreement"] == "agree"
        out["test"].append(ex)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"seed": SEED, "splits": {}}
    for name, examples in out.items():
        for ex in examples:
            ex["seq_tokens"] = seq_tokens(ex)
        lengths = sorted(ex["seq_tokens"] for ex in examples)
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        report["splits"][name] = {
            "examples": len(examples),
            "filings": len({(e["company"], e["year"]) for e in examples}),
            "companies": len({e["company"] for e in examples}),
            "seq_tokens_median": lengths[len(lengths) // 2], "seq_tokens_max": lengths[-1],
            "over_max_seq_len": sum(n > MAX_SEQ_LEN for n in lengths),
            "program_steps": dict(sorted(Counter(e["program"].count("(") for e in examples).items())),
            "yes_no": sum(isinstance(e["exe_ans"], str) for e in examples),
        }
        if name == "test":
            report["splits"][name]["label_consistent"] = sum(e["label_consistent"] for e in examples)

    filings = {n: {(e["company"], e["year"]) for e in out[n]} for n in ("sft_train", "sft_val", "pref_prompts")}
    report["filing_overlap"] = {f"{a}&{b}": len(filings[a] & filings[b])
                                for a, b in (("sft_train", "sft_val"), ("sft_train", "pref_prompts"),
                                             ("sft_val", "pref_prompts"))}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
