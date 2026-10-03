"""Pinned upstream sources. Every URL points at an immutable commit/revision so downloads are reproducible."""

FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
TATQA_REVISION = "c96247f5077eac447f63527fd3dcfdc58bb56d6a"

SOURCES = {
    "finqa": {
        "role": "training, preference construction, held-out in-domain eval",
        "homepage": "https://github.com/czyssrs/FinQA",
        "revision": FINQA_COMMIT,
        # GitHub repo is MIT; the HF mirror (ibm-research/finqa) lists CC BY 4.0. Attribute under both.
        "licenses": ["MIT (github.com/czyssrs/FinQA)", "CC-BY-4.0 (huggingface.co/datasets/ibm-research/finqa)"],
        "citation": "Chen et al., 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. EMNLP.",
        "files": {
            split: f"https://raw.githubusercontent.com/czyssrs/FinQA/{FINQA_COMMIT}/dataset/{split}.json"
            for split in ("train", "dev", "test")
        },
    },
    "tatqa": {
        "role": "external evaluation only (never used for training)",
        "homepage": "https://huggingface.co/datasets/next-tat/TAT-QA",
        "revision": TATQA_REVISION,
        "licenses": ["CC-BY-4.0"],
        "citation": "Zhu et al., 2021. TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance. ACL.",
        # No train split on purpose: TAT-QA is eval-only. test.json has no answers, so use test_gold.
        "files": {
            split: f"https://huggingface.co/datasets/next-tat/TAT-QA/resolve/{TATQA_REVISION}/tatqa_dataset_{name}.json"
            for split, name in (("dev", "dev"), ("test", "test_gold"))
        },
    },
}
