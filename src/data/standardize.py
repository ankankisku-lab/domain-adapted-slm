"""Phase 1: map raw FinQA / TAT-QA into one canonical, structured schema.

Nothing is dropped here: every raw record maps to exactly one standardized record, and structural problems are
counted in the report. Filtering is Phase 2's job, where every removal is logged.

Rendering into instruction/context/response text happens later (instruction construction), so the structured
fields (table rows, evidence, program) are preserved rather than flattened.

Usage:  python -m src.data.standardize
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/standard")
REPORT_PATH = Path("data/manifest/standardize_report.json")


class FinQARecord(TypedDict):
    id: str                   # upstream id, e.g. "ADI/2009/page_49.pdf-1"
    source: str               # "finqa"
    domain: str               # "financial_reasoning"
    original_split: str       # train | dev | test (upstream split, kept for provenance and test isolation)
    report_id: str            # report page, e.g. "ADI/2009/page_49.pdf"
    company: str | None       # ticker parsed from report_id
    year: int | None          # filing year parsed from report_id
    question: str
    pre_text: list[str]       # report sentences before the table
    table: list[list[str]]    # normalized table rows, first row is the header
    post_text: list[str]      # report sentences after the table
    gold_evidence: dict[str, str]  # {"text_3": "...", "table_1": "..."} supporting facts
    program: str              # flat DSL program, e.g. "subtract(112.4, 100), divide(#0, 100)"
    program_nested: str       # nested form of the same program
    exe_ans: float | str      # result of executing the program (upstream-computed)
    answer_text: str          # annotator answer string (may be empty or disagree with exe_ans)


class TATQARecord(TypedDict):
    id: str                   # question uid
    source: str               # "tatqa"
    domain: str
    original_split: str       # dev | test
    context_id: str           # table uid shared by the questions over the same table + paragraphs
    question: str
    table: list[list[str]]
    paragraphs: list[str]     # ordered paragraph texts
    answer: Any               # list[str] for span/multi-span, number for arithmetic/count
    answer_type: str          # span | multi-span | arithmetic | count
    answer_from: str          # table | text | table-text
    derivation: str           # arithmetic expression when answer_type == arithmetic
    scale: str                # "", thousand, million, billion, percent


def _parse_report_id(report_id: str) -> tuple[str | None, int | None]:
    parts = report_id.split("/")
    company = parts[0] if len(parts) >= 3 else None
    year = int(parts[1]) if len(parts) >= 3 and parts[1].isdigit() else None
    return company, year


def standardize_finqa(raw: dict, split: str) -> FinQARecord:
    qa = raw["qa"]
    company, year = _parse_report_id(raw["filename"])
    return {
        "id": raw["id"],
        "source": "finqa",
        "domain": "financial_reasoning",
        "original_split": split,
        "report_id": raw["filename"],
        "company": company,
        "year": year,
        "question": qa["question"],
        "pre_text": raw["pre_text"],
        "table": raw["table"],
        "post_text": raw["post_text"],
        "gold_evidence": qa["gold_inds"],
        "program": qa["program"],
        "program_nested": qa["program_re"],
        "exe_ans": qa["exe_ans"],
        "answer_text": qa["answer"],
    }


def standardize_tatqa(ctx: dict, split: str) -> list[TATQARecord]:
    paragraphs = [p["text"] for p in sorted(ctx["paragraphs"], key=lambda p: p["order"])]
    return [
        {
            "id": q["uid"],
            "source": "tatqa",
            "domain": "financial_reasoning",
            "original_split": split,
            "context_id": ctx["table"]["uid"],
            "question": q["question"],
            "table": ctx["table"]["table"],
            "paragraphs": paragraphs,
            "answer": q["answer"],
            "answer_type": q["answer_type"],
            "answer_from": q["answer_from"],
            "derivation": q["derivation"],
            "scale": q["scale"],
        }
        for q in sorted(ctx["questions"], key=lambda q: q["order"])
    ]


def structural_issues(rec: dict) -> list[str]:
    """Structural checks only; content-quality checks belong to Phase 2."""
    issues = []
    if not rec["question"].strip():
        issues.append("empty_question")
    if not rec["table"] or not any(any(cell.strip() for cell in row) for row in rec["table"]):
        issues.append("empty_table")
    if len({len(row) for row in rec["table"]}) > 1:
        issues.append("ragged_table")
    if rec["source"] == "finqa":
        if not rec["program"].strip():
            issues.append("empty_program")
        if not rec["gold_evidence"]:
            issues.append("no_gold_evidence")
        if not str(rec["answer_text"]).strip():
            issues.append("empty_answer_text")
        if isinstance(rec["exe_ans"], str):
            issues.append("non_numeric_exe_ans")  # e.g. yes/no answers from greater()
        if rec["company"] is None or rec["year"] is None:
            issues.append("unparsed_report_id")
    elif rec["answer"] in ("", [], None):
        issues.append("empty_answer")
    return issues


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    report: dict[str, Any] = {}
    outputs = {
        "finqa": [
            standardize_finqa(r, split)
            for split in ("train", "dev", "test")
            for r in json.loads((RAW_DIR / "finqa" / f"{split}.json").read_text(encoding="utf-8"))
        ],
        "tatqa_eval": [
            rec
            for split in ("dev", "test")
            for ctx in json.loads((RAW_DIR / "tatqa" / f"{split}.json").read_text(encoding="utf-8"))
            for rec in standardize_tatqa(ctx, split)
        ],
    }
    for name, records in outputs.items():
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        issues = Counter(i for r in records for i in structural_issues(r))
        report[name] = {
            "records": len(records),
            "by_split": dict(Counter(r["original_split"] for r in records)),
            "structural_issues": dict(issues),
        }
        _write_jsonl(OUT_DIR / f"{name}.jsonl", records)
        print(name, json.dumps(report[name]))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
