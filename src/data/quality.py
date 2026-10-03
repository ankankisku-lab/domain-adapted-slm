"""Phase 4: quality signals and filtering on the deduplicated pool, plus SFT response rendering.

Every signal is computed for every record and saved. Only signals that inspection showed to reliably indicate a
bad example become drop rules:
  truncated_question      question ends mid-phrase ("...shares of common stock of")
  question_year_mismatch  question names a year the report excerpt never mentions, and isn't a hypothetical
                          (all inspected cases were real mismatches, e.g. a 2013 report asked about 2018 sales)
Measured and reported but not used as filters (inspection showed they don't mark bad labels):
  length_ratio            response/question tokens. The low tail (< 0.5) is mostly long but valid questions with
                          short yes/no answers; the high tail is short questions over multi-step programs
                          ("what was the average X"). Neither indicates a bad example in FinQA.
  repetitive_context      repeated PDF boilerplate sentences; the question and label are still fine

Usage:  python -m src.data.quality
"""

import json
import re
from collections import Counter
from pathlib import Path

import src.paths  # noqa: F401  (keeps the HF cache on D:)
from src.data.render import render_prompt, render_response

IN_PATH = Path("data/interim/finqa_dedup.jsonl")
RAW_PATH = Path("data/standard/finqa.jsonl")
OUT_PATH = Path("data/interim/finqa_quality.jsonl")
REPORT_PATH = Path("data/manifest/quality_report.json")
TOKENIZER_REPO = "unsloth/Llama-3.2-3B-Instruct"

LOW_LENGTH_RATIO = 0.5  # reported only
DROP_RULES = ["truncated_question", "question_year_mismatch"]

_YEAR = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
_SHORT_YEAR = re.compile(r"(?:'|/)(\d{2})\b")
_HYPOTHETICAL = re.compile(r"\b(would|will|expect\w*|assum\w*|if|project\w*|forecast\w*|estimat\w*|anticipat\w*|next)\b")
_TRUNCATED_TAIL = {"of", "the", "a", "an", "and", "or", "with", "by", "was", "were", "is", "are"}


def _context_text(rec: dict) -> str:
    return " ".join(rec["pre_text"] + rec["post_text"] + [c for row in rec["table"] for c in row]
                    + list(rec["gold_evidence"].values()))


def signals(rec: dict, response: str, tokenizer) -> dict:
    q = rec["question"].strip()
    words = q.rstrip("?").split()
    tail = words[-1].lower() if words else ""
    context = _context_text(rec)
    context_years = {"20" + y for y in _SHORT_YEAR.findall(context)}
    missing_years = [y for y in _YEAR.findall(q) if y not in context and y not in context_years]
    sentences = [s for s in rec["pre_text"] + rec["post_text"] if len(s) > 30]
    q_tokens = len(tokenizer.encode(q).ids)
    r_tokens = len(tokenizer.encode(response).ids)
    return {
        "truncated_question": tail in _TRUNCATED_TAIL and not q.lower().rstrip("?").endswith("series a"),
        "question_year_mismatch": bool(missing_years) and not _HYPOTHETICAL.search(q.lower()),
        "missing_years": missing_years,
        "question_tokens": q_tokens,
        "response_tokens": r_tokens,
        "prompt_tokens": len(tokenizer.encode(render_prompt(rec)).ids),
        "length_ratio": round(r_tokens / max(1, q_tokens), 3),
        "low_length_ratio": r_tokens / max(1, q_tokens) < LOW_LENGTH_RATIO,
        "repetitive_context": bool(sentences) and len(set(sentences)) < 0.8 * len(sentences),
        "program_steps": rec["program"].count("(") - rec["program"].count("((") if rec["program"] else 0,
    }


def main() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    raw_tables = {}
    for line in RAW_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        raw_tables[r["id"]] = r["table"]

    rows = [json.loads(line) for line in IN_PATH.open(encoding="utf-8")]
    funnel = {"pool_in": len(rows), **{f"dropped_{d}": 0 for d in DROP_RULES}, "pool_out": 0}
    flagged: Counter = Counter()
    kept = []
    for rec in rows:
        response = render_response(rec, raw_tables[rec["id"]])
        sig = signals(rec, response, tokenizer)
        flagged.update(k for k in ("truncated_question", "question_year_mismatch", "low_length_ratio",
                                   "repetitive_context") if sig[k])
        failed = [d for d in DROP_RULES if sig[d]]
        if failed:
            funnel[f"dropped_{failed[0]}"] += 1
            continue
        rec["response"] = response
        rec["quality"] = sig
        kept.append(rec)
    funnel["pool_out"] = len(kept)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report = {"config": {"low_length_ratio_reported_below": LOW_LENGTH_RATIO, "drop_rules": DROP_RULES},
              "funnel": funnel, "flagged_non_exclusive": dict(flagged)}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
