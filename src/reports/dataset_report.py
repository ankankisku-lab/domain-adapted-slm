"""Phase 5b: self-contained HTML dataset report (charts + tables) from the pipeline manifests and final splits.

Usage:  python -m src.reports.dataset_report   ->  reports/dataset_report.html
"""

import datetime
import json
from pathlib import Path

MANIFEST = Path("data/manifest")
FINAL = Path("data/final")
OUT_PATH = Path("reports/dataset_report.html")
MAX_SEQ_LEN = 2048
BIN = 128


def _load(name: str) -> dict:
    return json.loads((MANIFEST / name).read_text(encoding="utf-8"))


def build_data() -> dict:
    cleaning, dedup, quality, splits = (_load(n) for n in (
        "cleaning_report.json", "dedup_report.json", "quality_report.json", "splits_report.json"))
    labels = json.loads(Path("data/labels/dedup_labels_provenance.json").read_text(encoding="utf-8"))
    f2 = cleaning["pool_funnel_train_dev"]

    funnel = [
        {"stage": "Train + dev (raw)", "value": f2["pool_in"]},
        {"stage": "Cleaned + verified", "value": f2["pool_out"]},
        {"stage": "Deduplicated", "value": dedup["pool_out"]},
        {"stage": "Quality-filtered", "value": quality["funnel"]["pool_out"]},
    ]
    drops = [
        ["Phase 2", "Program result differs from recorded exe_ans", f2["dropped_program_executes"]],
        ["Phase 2", "Annotator answer disagrees with or is missing from the executed result", f2["dropped_answer_agreement"]],
        ["Phase 2", "Program uses a number not found in the context", f2["dropped_grounding"]],
        ["Phase 2", f"Prompt longer than {cleaning['config']['max_prompt_tokens']} tokens", f2["dropped_length"]],
        ["Phase 3", "Near-duplicate of another training example", dedup["removed_duplicates"]],
        ["Phase 3", "Near-duplicate of a test example (contamination)", dedup["removed_test_contamination"]],
        ["Phase 4", "Truncated question", quality["funnel"]["dropped_truncated_question"]],
        ["Phase 4", "Question names a year absent from the report excerpt", quality["funnel"]["dropped_question_year_mismatch"]],
    ]

    hist = []
    train_lengths = [json.loads(l)["seq_tokens"] for l in (FINAL / "sft_train.jsonl").open(encoding="utf-8")]
    for lo in range(0, MAX_SEQ_LEN, BIN):
        hist.append({"lo": lo, "hi": lo + BIN, "count": sum(lo <= n < lo + BIN for n in train_lengths)})

    table = dedup["detector_table"]
    rules = sorted({r["rule"] for r in table}, key=lambda r: len(r))
    series = [{"name": rule, "points": [{"x": r["threshold"], "y": r["f1"], "p": r["precision"], "r": r["recall"]}
                                        for r in table if r["rule"] == rule]} for rule in rules]

    split_rows = []
    for name in ("sft_train", "sft_val", "pref_prompts", "test"):
        s = splits["splits"][name]
        steps = s["program_steps"]
        split_rows.append([name, s["examples"], s["filings"], s["companies"], s["seq_tokens_median"],
                           s["seq_tokens_max"], steps.get("1", 0), steps.get("2", 0),
                           sum(v for k, v in steps.items() if int(k) >= 3), s["yes_no"]])

    return {
        "generated": datetime.date.today().isoformat(),
        "tiles": [
            ["Verified pool", quality["funnel"]["pool_out"]],
            ["SFT train", splits["splits"]["sft_train"]["examples"]],
            ["ORPO prompts", splits["splits"]["pref_prompts"]["examples"]],
            ["SFT validation", splits["splits"]["sft_val"]["examples"]],
            ["FinQA test (untouched)", splits["splits"]["test"]["examples"]],
        ],
        "funnel": funnel, "drops": drops, "hist": hist, "maxSeq": MAX_SEQ_LEN,
        "dedup": {"series": series, "chosen": dedup["chosen"], "labeled": dedup["labeled_pairs"],
                  "positives": dedup["labeled_duplicates"], "annotator": labels["annotator"]},
        "splits": split_rows, "overlap": splits["filing_overlap"],
        "testConsistent": splits["splits"]["test"]["label_consistent"],
        "rules": sorted(cleaning["cleaning_rule_applications"].items(), key=lambda kv: -kv[1]),
    }


def main() -> None:
    data = build_data()
    template = (Path(__file__).parent / "dataset_report_template.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
