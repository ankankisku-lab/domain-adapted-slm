"""Re-score saved predictions with the current scorer (no GPU needed) and rewrite the metrics file.

Keeps every model's numbers on the same scorer version after a scoring change. The original "run" block
(model, GPU, decoding, etc.) is preserved.

Usage:  python -m src.eval.rescore results/baseline_test_predictions.jsonl
"""

import json
import sys
from pathlib import Path

from src.eval.run_eval import load_split, summarize
from src.eval.scoring import score_record


def main(pred_path: str) -> None:
    path = Path(pred_path)
    tag_split = path.name.removesuffix("_predictions.jsonl")
    split = next(s for s in ("sft_val", "pref_prompts", "tatqa_test", "test") if tag_split.endswith("_" + s))
    rows = load_split(split, None)
    by_id = {r["id"]: r for r in rows}
    preds = {}
    changed = 0
    for line in path.open(encoding="utf-8"):
        p = json.loads(line)
        r = by_id[p["id"]]
        correct, extracted = score_record(r, p["output"])
        changed += correct != p["correct"]
        p.update(correct=correct, extracted=extracted)
        preds[p["id"]] = p
    path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds.values()), encoding="utf-8")
    metrics_path = path.with_name(f"{tag_split}_metrics.json")
    old = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics = summarize(rows, preds)
    metrics["run"] = old.get("run", {})
    metrics["rescored_changes"] = changed
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "run"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
