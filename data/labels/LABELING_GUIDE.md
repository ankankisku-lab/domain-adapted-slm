# Duplicate labeling guide (Phase 3)

File: `dedup_pairs_to_label.csv`. Fill the `is_duplicate (1/0)` column for all 200 rows, then save it as CSV with the same name.

Similarity scores are hidden and rows are shuffled on purpose, so judge only what's in front of you.

## The question to ask

> If the model trained on A, would training on B teach it almost nothing new?

Mark **1 (duplicate)** when both of these hold:
- A and B ask the **same thing**, even if the wording differs ("what percentage of X was Y" vs "what portion of X is Y").
- They rest on the **same facts**, so the answer and the calculation would be the same.

Mark **0 (not a duplicate)** when any of these is true:
- A different **year, period, company or line item** is involved ("net change during 2012" vs "during 2011").
- The same question template is applied to **different numbers**. The answers will differ.
- The facts overlap but the **calculation differs** (a total vs an average of the same rows).

## Tips
- Compare the `a_answer` / `b_answer` columns first. Different answers almost always mean 0.
- Same answers don't automatically mean 1. Check that the question intent matches too.
- If a pair is genuinely ambiguous, pick your best guess and move on. A handful of hard cases won't change the chosen threshold.
- Expect about 10–20 minutes for all 200 rows.
