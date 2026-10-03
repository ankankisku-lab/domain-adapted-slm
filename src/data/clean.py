"""Phase 2: clean FinQA and verify labels.

Train + dev form the training pool: they're normalized, checked, and filtered. Test is only normalized (the same input
preprocessing the model sees at train time) and flagged, never filtered, so it stays comparable to published FinQA
results.

Checks, in funnel order. A dropped record is attributed to the first check it fails:
  1. structural          empty question/table/program
  2. number_preservation cleaning must not lose any number from the context (invariant; failure = cleaning bug)
  3. program_executes    our DSL executor must reproduce FinQA's recorded exe_ans
  4. answer_agreement    the annotator answer must agree with the executed result (label consistency)
  5. grounding           every literal number in the program must appear in the context or question
  6. length              rendered context + question must fit the prompt token budget

Usage:  python -m src.data.clean
"""

import json
from collections import Counter
from pathlib import Path

import src.paths  # noqa: F401  (keeps the HF cache on D:)
from src.data.finqa_program import answer_agrees, execute, parse_program, same_result
from src.data.render import render_context
from src.data.text_clean import clean_sentences, clean_table, clean_text, decode_artifacts, numbers_in

IN_PATH = Path("data/standard/finqa.jsonl")
OUT_PATH = Path("data/interim/finqa_clean.jsonl")
REPORT_PATH = Path("data/manifest/cleaning_report.json")

TOKENIZER_REPO = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LEN = 2048
RESPONSE_BUDGET = 256        # tokens reserved for the answer with its reasoning steps
PROMPT_OVERHEAD = 64         # chat template + instruction wrapper, re-measured once the SFT template is fixed
MAX_PROMPT_TOKENS = MAX_SEQ_LEN - RESPONSE_BUDGET - PROMPT_OVERHEAD

CHECKS = ["structural", "number_preservation", "program_executes", "answer_agreement", "grounding", "length"]


def _context_numbers(rec: dict, decode: bool = False) -> set[float]:
    """`decode=True` for raw records: code points like "2019" in "company 2019s" are punctuation, not numbers."""
    parts = rec["pre_text"] + rec["post_text"] + [c for row in rec["table"] for c in row]
    return numbers_in(" ".join(decode_artifacts(p) if decode else p for p in parts))


def _program_literals(program: str) -> list[float]:
    values = []
    for _, *args in parse_program(program) or []:
        for a in args:
            if a.startswith(("#", "const_")) or a == "none":
                continue
            try:
                values.append(abs(float(a.replace(",", "").rstrip("%"))))
            except ValueError:
                pass  # row name for table ops
    return values


def clean_record(raw: dict, rule_counts: Counter) -> dict:
    rec = dict(raw)
    rec["question"] = clean_text(raw["question"], rule_counts)
    rec["pre_text"] = clean_sentences(raw["pre_text"], rule_counts)
    rec["post_text"] = clean_sentences(raw["post_text"], rule_counts)
    rec["table"] = clean_table(raw["table"], rule_counts)
    rec["gold_evidence"] = {k: clean_text(v) for k, v in raw["gold_evidence"].items()}
    return rec


def run_checks(raw: dict, rec: dict, tokenizer) -> tuple[dict[str, bool], dict]:
    """Returns pass/fail per check (True = pass) and measured values worth keeping."""
    passed, info = {}, {}
    passed["structural"] = bool(rec["question"].strip() and rec["table"] and raw["program"].strip())

    lost = _context_numbers(raw, decode=True) - _context_numbers(rec)
    passed["number_preservation"] = not lost
    if lost:
        info["numbers_lost"] = sorted(lost)[:10]

    # Execute on the raw table: program row-name arguments are written against the uncleaned row labels.
    ours = execute(raw["program"], raw["table"])
    passed["program_executes"] = ours is not None and same_result(ours, raw["exe_ans"])

    agrees = answer_agrees(raw["answer_text"], raw["exe_ans"])
    passed["answer_agreement"] = agrees is True
    info["answer_agreement"] = {True: "agree", False: "disagree", None: "missing"}[agrees]

    available = _context_numbers(rec) | numbers_in(rec["question"])
    missing = [v for v in _program_literals(raw["program"])
               if not any(abs(v - a) <= 1e-9 + 1e-6 * a for a in available)]
    passed["grounding"] = not missing
    if missing:
        info["ungrounded_args"] = missing

    prompt_tokens = len(tokenizer.encode(render_context(rec) + "\n\nQuestion: " + rec["question"]).ids)
    info["prompt_tokens"] = prompt_tokens
    passed["length"] = prompt_tokens <= MAX_PROMPT_TOKENS
    return passed, info


def main() -> None:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    raws = [json.loads(line) for line in IN_PATH.open(encoding="utf-8")]

    rule_counts: Counter = Counter()
    funnel = {"pool_in": 0, **{f"dropped_{c}": 0 for c in CHECKS}, "pool_out": 0}
    failures: dict[str, Counter] = {"pool": Counter(), "test": Counter()}
    answer_status: dict[str, Counter] = {"pool": Counter(), "test": Counter()}
    kept_by_split: Counter = Counter()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for raw in raws:
            rec = clean_record(raw, rule_counts)
            passed, info = run_checks(raw, rec, tokenizer)
            group = "test" if raw["original_split"] == "test" else "pool"
            failed = [c for c in CHECKS if not passed[c]]
            failures[group].update(failed)
            answer_status[group][info["answer_agreement"]] += 1

            if group == "test":
                keep = True
            else:
                funnel["pool_in"] += 1
                keep = not failed
                if failed:
                    funnel[f"dropped_{failed[0]}"] += 1
                else:
                    funnel["pool_out"] += 1
            if keep:
                kept_by_split[raw["original_split"]] += 1

            rec.update({"failed_checks": failed, "keep": keep, "check_info": info})
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "config": {"max_seq_len": MAX_SEQ_LEN, "response_budget": RESPONSE_BUDGET,
                   "prompt_overhead": PROMPT_OVERHEAD, "max_prompt_tokens": MAX_PROMPT_TOKENS,
                   "tokenizer": TOKENIZER_REPO},
        "pool_funnel_train_dev": funnel,
        "kept_by_split": dict(kept_by_split),
        "check_failures_non_exclusive": {g: dict(c) for g, c in failures.items()},
        "answer_agreement": {g: dict(c) for g, c in answer_status.items()},
        "cleaning_rule_applications": dict(rule_counts.most_common()),
        "html_or_non_ascii_found": 0,  # verified in the Phase 2 survey; NFKC normalization still applied
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
