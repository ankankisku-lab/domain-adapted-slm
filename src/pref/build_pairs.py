"""Phase 9b: turn scored SFT samples into ORPO preference pairs (runs locally, no GPU).

Rubric, in order (correctness is decided by the shared scorer against the program-executed gold answer):
  1. rejected = a sampled answer the scorer marks wrong. Answers that still reach an "Answer:" line are preferred:
     plausible-but-wrong is the mistake worth unlearning.
  2. chosen = a sampled answer the scorer marks right (on-policy). If no sample was right, the verified reference
     response (the rendered gold program) is used instead, and the pair is tagged chosen_source="reference".
  3. prompts where every sample was right give no pair: there is nothing to prefer.
  4. one pair per prompt; if that leaves fewer than --target pairs, prompts with another distinct wrong sample
     contribute a second pair until the target is reached.
Identical outputs are deduplicated before pairing. Every pair is re-scored as a check: chosen right, rejected wrong.

Output: data/final/pref_pairs.jsonl in TRL's conversational preference format
  {"prompt": [system, user], "chosen": [assistant], "rejected": [assistant], ...metadata}

Usage:  python -m src.pref.build_pairs --samples results/sft_r16_pref_prompts_samples.jsonl
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.eval.run_eval import load_split
from src.eval.scoring import is_correct

OUT_PATH = Path("data/final/pref_pairs.jsonl")
REPORT_PATH = Path("data/manifest/pref_pairs_report.json")


def _dedupe(samples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in samples:
        key = s["output"].strip()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--target", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = {r["id"]: r for r in load_split("pref_prompts", None)}
    sampled = [json.loads(line) for line in Path(args.samples).open(encoding="utf-8")]

    stats: Counter = Counter()
    first, second = [], []
    for s in sampled:
        r = rows[s["id"]]
        uniq = _dedupe(s["samples"])
        right = [x for x in uniq if x["correct"]]
        wrong = [x for x in uniq if not x["correct"]]
        stats["prompts"] += 1
        stats[f"correct_{sum(x['correct'] for x in s['samples'])}_of_{s['k']}"] += 1
        if not wrong:
            stats["no_pair_all_correct"] += 1
            continue
        # Hard negatives first: wrong answers that still commit to an "Answer:" line.
        wrong.sort(key=lambda x: ("answer:" not in x["output"].lower(), rng.random()))
        rng.shuffle(right)
        chosen_pool = right or [{"output": r["messages"][2]["content"], "extracted": r["answer"]}]
        source = "sampled" if right else "reference"
        for i, rej in enumerate(wrong[:2]):
            ch = chosen_pool[i % len(chosen_pool)]
            pair = {
                "id": f"{r['id']}#{i}", "source_id": r["id"], "prompt": r["messages"][:2],
                "chosen": [{"role": "assistant", "content": ch["output"]}],
                "rejected": [{"role": "assistant", "content": rej["output"]}],
                "chosen_source": source, "gold": r["answer"],
                "chosen_extracted": ch["extracted"], "rejected_extracted": rej["extracted"],
                "rejected_has_answer_line": "answer:" in rej["output"].lower(),
            }
            (first if i == 0 else second).append(pair)

    pairs = first + second[:max(0, args.target - len(first))]
    for p in pairs:  # rubric check on the final set
        src = rows[p["source_id"]]
        assert is_correct(p["chosen"][0]["content"], src["exe_ans"], src["question"])[0], p["id"]
        assert not is_correct(p["rejected"][0]["content"], src["exe_ans"], src["question"])[0], p["id"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    report = {
        "samples_file": args.samples, "target": args.target, "pairs": len(pairs),
        "prompts_with_pairs": len({p["source_id"] for p in pairs}),
        "second_pairs_used": len(pairs) - len(first) if len(pairs) > len(first) else 0,
        "chosen_source": dict(Counter(p["chosen_source"] for p in pairs)),
        "rejected_has_answer_line": sum(p["rejected_has_answer_line"] for p in pairs),
        "prompt_outcomes": dict(sorted(stats.items())),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
