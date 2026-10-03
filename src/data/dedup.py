"""Phase 3: semantic deduplication with BGE-small, with a human-calibrated similarity threshold.

Unit of comparison: question + gold evidence sentences. Question-only embeddings would merge FinQA's templated questions
("percentage change in X from 2013 to 2014") asked about different facts, which are not duplicates.

Steps:
  embed   encode the training pool (kept train+dev) and the test set, cache vectors
  sheet   find nearest-neighbour pairs, sample ~200 across similarity bins, write a blind labeling sheet
  apply   (after labeling) estimate precision/recall per threshold, choose one, remove duplicates and test-contaminated
          pool items

Search is exact (normalized dot products with numpy). At ~7.5K vectors that's cheaper than building a FAISS index and has
no approximation error.

Usage:  python -m src.data.dedup embed | sheet | apply [--threshold T]
"""

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

import src.paths  # noqa: F401  (keeps model caches on D:)
from src.data.clean import _program_literals
from src.data.finqa_program import same_result

CLEAN_PATH = Path("data/interim/finqa_clean.jsonl")
EMB_PATH = Path("data/interim/dedup_embeddings.npz")
SHEET_PATH = Path("data/labels/dedup_pairs_to_label.csv")
KEY_PATH = Path("data/labels/dedup_pairs_key.json")
MODEL_ID = "BAAI/bge-small-en-v1.5"

TOP_K = 5
# BGE-small similarities bunch near the top: over half of all nearest-neighbour pairs score >= 0.90, and even >= 0.99
# includes non-duplicates that differ only by year. Pairs whose executed answers match are rare (~420 of ~26K) but are
# where true duplicates live, so each similarity bin is sampled in two strata: same answer / different answer.
# Labels are re-weighted by stratum population when precision/recall are estimated. Pairs below 0.80 aren't sampled
# (assumed duplicate-free: only 9 same-answer pairs exist even in 0.80-0.88).
SAMPLING_PLAN = [  # (low, high, same-answer pairs, different-answer pairs)
    (0.80, 0.88, 6, 10), (0.88, 0.92, 10, 12), (0.92, 0.95, 16, 14), (0.95, 0.97, 17, 16),
    (0.97, 0.98, 17, 16), (0.98, 0.99, 17, 16), (0.99, 1.0001, 17, 16),
]
SEED = 13


def load_records() -> list[dict]:
    """Pool items (kept train/dev) and all test items; test is only a contamination reference."""
    rows = [json.loads(line) for line in CLEAN_PATH.open(encoding="utf-8")]
    return [r for r in rows if r["keep"]]


def dedup_text(rec: dict) -> str:
    evidence = " ".join(rec["gold_evidence"].values())
    return f"question: {rec['question']}\nevidence: {evidence}"


def exact_key(rec: dict) -> str:
    return re.sub(r"\s+", " ", dedup_text(rec).lower()).strip()


def cmd_embed() -> None:
    from sentence_transformers import SentenceTransformer

    recs = load_records()
    model = SentenceTransformer(MODEL_ID, device="cpu")
    vecs = model.encode([dedup_text(r) for r in recs], batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMB_PATH, vecs=vecs, ids=np.array([r["id"] for r in recs]))
    print(f"embedded {len(recs)} records -> {EMB_PATH} {vecs.shape}")


def neighbour_pairs(vecs: np.ndarray, is_test: np.ndarray, k: int = TOP_K) -> dict[tuple[int, int], float]:
    """Top-k neighbours of every item; keep pairs that involve at least one pool item."""
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1.0)
    top = np.argpartition(-sims, k, axis=1)[:, :k]
    pairs = {}
    for i in range(len(vecs)):
        for j in top[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            if is_test[a] and is_test[b]:
                continue
            pairs[(a, b)] = float(sims[a, b])
    return pairs


def _pair_kind(is_test: np.ndarray, a: int, b: int) -> str:
    return "pool-test" if is_test[a] or is_test[b] else "pool-pool"


def cmd_sheet() -> None:
    recs = load_records()
    data = np.load(EMB_PATH)
    assert list(data["ids"]) == [r["id"] for r in recs], "embeddings are stale; rerun `embed`"
    vecs = data["vecs"]
    is_test = np.array([r["original_split"] == "test" for r in recs])

    keys = [exact_key(r) for r in recs]
    exact_groups = Counter(k for k, t in zip(keys, is_test) if not t)
    pairs = neighbour_pairs(vecs, is_test)
    # Identical text is handled deterministically; the sheet calibrates only the fuzzy region.
    fuzzy = {p: s for p, s in pairs.items() if keys[p[0]] != keys[p[1]]}

    same = {p: same_result(recs[p[0]]["exe_ans"], recs[p[1]]["exe_ans"]) for p in fuzzy}
    rng = random.Random(SEED)
    sampled, strata = [], {}
    for lo, hi, n_same, n_diff in SAMPLING_PLAN:
        for is_same, n in ((True, n_same), (False, n_diff)):
            members = [p for p, s in fuzzy.items() if lo <= s < hi and same[p] == is_same]
            name = f"{lo:.2f}-{min(hi, 1.0):.2f}|{'same' if is_same else 'diff'}"
            picked = rng.sample(members, min(n, len(members)))
            strata[name] = {"population": len(members), "sampled": len(picked)}
            sampled += [(p, name) for p in picked]
    rng.shuffle(sampled)  # blind: the labeler never sees similarity, stratum or order

    SHEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = {}
    with SHEET_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "is_duplicate (1/0)", "a_question", "b_question", "a_answer", "b_answer",
                    "a_evidence", "b_evidence", "a_report", "b_report"])
        for n, ((a, b), stratum) in enumerate(sampled, 1):
            ra, rb = recs[a], recs[b]
            w.writerow([n, "", ra["question"], rb["question"], ra["answer_text"], rb["answer_text"],
                        " ".join(ra["gold_evidence"].values()), " ".join(rb["gold_evidence"].values()),
                        ra["report_id"], rb["report_id"]])
            key[n] = {"a": ra["id"], "b": rb["id"], "similarity": fuzzy[(a, b)], "kind": _pair_kind(is_test, a, b),
                      "same_exe_ans": same[(a, b)], "stratum": stratum}

    KEY_PATH.write_text(json.dumps({"strata": strata, "pairs": key}, indent=2), encoding="utf-8")
    sims = np.array(list(pairs.values()))
    print(json.dumps({
        "pool_items": int((~is_test).sum()), "test_items": int(is_test.sum()),
        "exact_duplicate_extra_copies_in_pool": sum(c - 1 for c in exact_groups.values() if c > 1),
        "neighbour_pairs": len(pairs), "pairs_sim>=0.90": int((sims >= 0.90).sum()),
        "strata": strata, "sheet_rows": len(sampled),
    }, indent=2))


THRESHOLDS = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
# Candidate duplicate rules, least to most strict. "same_args" = the programs use the same set of literal numbers,
# i.e. the same calculation inputs. It separates true duplicates from coincidentally equal answers computed from
# different facts (e.g. a 20% junk-rating share in two different years).
RULES = ("similarity", "similarity+same_answer", "similarity+same_answer+same_args")


def program_args(rec: dict) -> frozenset:
    return frozenset(round(v, 6) for v in _program_literals(rec["program"]))


def rule_fires(rule: str, similarity: float, threshold: float, same_answer: bool, same_args: bool) -> bool:
    if similarity < threshold:
        return False
    if rule == "similarity":
        return True
    if rule == "similarity+same_answer":
        return same_answer
    return same_answer and same_args
OUT_PATH = Path("data/interim/finqa_dedup.jsonl")
REPORT_PATH = Path("data/manifest/dedup_report.json")


def load_labels() -> list[dict]:
    """Labeled pairs with similarity recomputed from the current embeddings (text cleaning may have changed since
    the sheet was drawn). Stratum weights keep the original sampling design."""
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    recs = load_records()
    by_id = {r["id"]: r for r in recs}
    data = np.load(EMB_PATH)
    assert list(data["ids"]) == [r["id"] for r in recs], "embeddings are stale; rerun `embed`"
    vec_of = dict(zip(data["ids"], data["vecs"]))
    with SHEET_PATH.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labeled, missing, dropped_since = [], [], 0
    for row in rows:
        value = row["is_duplicate (1/0)"].strip()
        if value not in ("0", "1"):
            missing.append(row["pair_id"])
            continue
        pair = dict(key["pairs"][row["pair_id"]])
        if pair["a"] not in by_id or pair["b"] not in by_id:
            dropped_since += 1  # record no longer in the pool (re-cleaning changed its checks)
            continue
        pair["similarity"] = float(vec_of[pair["a"]] @ vec_of[pair["b"]])
        stratum = key["strata"][pair["stratum"]]
        labeled.append({**pair, "is_dup": value == "1", "weight": stratum["population"] / stratum["sampled"],
                        "same_args": program_args(by_id[pair["a"]]) == program_args(by_id[pair["b"]])})
    if missing:
        raise SystemExit(f"{len(missing)} rows are unlabeled or not 0/1, e.g. pair_ids {missing[:10]}")
    if dropped_since:
        print(f"note: {dropped_since} labeled pairs skipped because a record left the pool")
    return labeled


def evaluate_detectors(labeled: list[dict]) -> list[dict]:
    """Population-weighted precision/recall/F1 for each (rule, threshold)."""
    results = []
    for rule in RULES:
        for t in THRESHOLDS:
            tp = fp = fn = 0.0
            for p in labeled:
                pred = rule_fires(rule, p["similarity"], t, p["same_exe_ans"], p["same_args"])
                tp += p["weight"] * (pred and p["is_dup"])
                fp += p["weight"] * (pred and not p["is_dup"])
                fn += p["weight"] * (not pred and p["is_dup"])
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            results.append({"rule": rule, "threshold": t, "precision": round(precision, 3),
                            "recall": round(recall, 3), "f1": round(f1, 3)})
    return results


def _components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def cmd_apply(threshold: float | None, rule: str | None) -> None:
    labeled = load_labels()
    table = evaluate_detectors(labeled)
    if threshold is None or rule is None:
        # Best F1; ties go to the stricter rule, then the higher threshold (fewer false removals).
        best = max(table, key=lambda r: (r["f1"], RULES.index(r["rule"]), r["threshold"]))
        rule, threshold = best["rule"], best["threshold"]

    recs = load_records()
    vecs = np.load(EMB_PATH)["vecs"]
    is_test = np.array([r["original_split"] == "test" for r in recs])
    keys = [exact_key(r) for r in recs]
    pairs = neighbour_pairs(vecs, is_test)

    def is_dup(a: int, b: int, sim: float) -> bool:
        if keys[a] == keys[b]:
            return True
        same = same_result(recs[a]["exe_ans"], recs[b]["exe_ans"])
        return rule_fires(rule, sim, threshold, same, program_args(recs[a]) == program_args(recs[b]))

    edges = [(a, b) for (a, b), s in pairs.items() if is_dup(a, b, s)]
    by_key: dict[str, list[int]] = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    edges += [(g[0], j) for g in by_key.values() for j in g[1:]]  # exact duplicates even if not top-k neighbours

    contaminated = {i for a, b in edges if is_test[a] != is_test[b] for i in (a, b) if not is_test[i]}
    pool_edges = [(a, b) for a, b in edges if not is_test[a] and not is_test[b]]
    pool_idx = [i for i in range(len(recs)) if not is_test[i]]
    removed_dup = set()
    for comp in _components(len(recs), pool_edges):
        comp = [i for i in comp if not is_test[i] and i not in contaminated]
        removed_dup.update(sorted(comp, key=lambda i: recs[i]["id"])[1:])  # keep one per duplicate group

    kept = [recs[i] for i in pool_idx if i not in contaminated and i not in removed_dup]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_exact = sum(len(g) - 1 for g in by_key.values() if len(g) > 1 and not any(is_test[i] for i in g))
    report = {
        "labeled_pairs": len(labeled), "labeled_duplicates": sum(p["is_dup"] for p in labeled),
        "detector_table": table, "chosen": {"rule": rule, "threshold": threshold},
        "pool_in": len(pool_idx), "removed_test_contamination": len(contaminated),
        "removed_duplicates": len(removed_dup), "of_which_exact_copies_approx": n_exact, "pool_out": len(kept),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["embed", "sheet", "apply"])
    parser.add_argument("--threshold", type=float, help="override the calibrated threshold")
    parser.add_argument("--rule", choices=RULES, help="override the rule")
    args = parser.parse_args()
    if args.command == "embed":
        cmd_embed()
    elif args.command == "sheet":
        cmd_sheet()
    else:
        cmd_apply(args.threshold, args.rule)


if __name__ == "__main__":
    main()
