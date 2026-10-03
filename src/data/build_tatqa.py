"""External evaluation set: a fixed, stratified 500-question subset of TAT-QA test, rendered like FinQA prompts.

TAT-QA (Zhu et al., 2021) is never trained on. It checks whether gains transfer to different companies, a different
annotation style and answer types FinQA doesn't have (text spans, multi-span lists, counts). The subset is stratified
by answer type with a fixed seed, so every model is scored on the same 500 questions.

Usage:  python -m src.data.build_tatqa   ->  data/final/tatqa_test.jsonl
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.data.render import SYSTEM_PROMPT, render_table

IN_PATH = Path("data/standard/tatqa_eval.jsonl")
OUT_PATH = Path("data/final/tatqa_test.jsonl")
SUBSET = 500
SEED = 20260927


def render_prompt(rec: dict) -> str:
    parts = [render_table(rec["table"]), "\n\n".join(rec["paragraphs"])]
    return "\n\n".join(p for p in parts if p.strip()) + f"\n\nQuestion: {rec['question']}"


def display_answer(rec: dict) -> str:
    a = rec["answer"]
    text = ", ".join(a) if isinstance(a, list) else str(a)
    suffix = {"percent": "%", "thousand": " thousand", "million": " million", "billion": " billion"}.get(rec["scale"], "")
    return text + suffix


def main() -> None:
    rows = [json.loads(line) for line in IN_PATH.open(encoding="utf-8")]
    rows = [r for r in rows if r["original_split"] == "test"]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["answer_type"]].append(r)
    rng = random.Random(SEED)
    picked = []
    for t, group in sorted(by_type.items()):
        n = round(SUBSET * len(group) / len(rows))
        picked += rng.sample(group, n)
    picked = picked[:SUBSET]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in picked:
            answer = display_answer(r)
            f.write(json.dumps({
                "id": r["id"], "dataset": "tatqa", "question": r["question"], "program": "",
                "exe_ans": r["answer"], "answer": answer, "answer_type": r["answer_type"],
                "answer_from": r["answer_from"], "scale": r["scale"],
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": render_prompt(r)},
                             {"role": "assistant", "content": f"Answer: {answer}"}],
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"questions": len(picked), "by_answer_type": dict(Counter(r["answer_type"] for r in picked)),
                      "by_scale": dict(Counter(r["scale"] or "none" for r in picked))}, indent=2))


if __name__ == "__main__":
    main()
