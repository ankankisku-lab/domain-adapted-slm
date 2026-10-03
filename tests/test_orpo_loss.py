"""CPU tests for the ORPO objective and pair data path (python -m tests.test_orpo_loss)."""

import math

import torch
import torch.nn.functional as F

from src.train.data import IGNORE
from src.train.orpo import PairCollator, orpo_loss, response_logps


def naive_logps(logits, labels):
    out = []
    for b in range(labels.shape[0]):
        lp, n = 0.0, 0
        for t in range(1, labels.shape[1]):
            if labels[b, t] != IGNORE:
                lp += F.log_softmax(logits[b, t - 1].double(), -1)[labels[b, t]].item()
                n += 1
        out.append(lp / max(n, 1))
    return torch.tensor(out, dtype=torch.float64)


def test_response_logps_matches_naive():
    torch.manual_seed(0)
    logits = torch.randn(4, 12, 50)
    labels = torch.randint(0, 50, (4, 12))
    labels[:, :5] = IGNORE          # prompt
    labels[1, 9:] = IGNORE          # padding on one row
    got, counts = response_logps(logits, labels)
    assert torch.allclose(got.double(), naive_logps(logits, labels), atol=1e-5)
    assert counts.tolist() == [7, 4, 7, 7]


def test_orpo_loss_matches_formula():
    torch.manual_seed(1)
    logits = torch.randn(4, 10, 30, requires_grad=True)
    labels = torch.randint(0, 30, (4, 10))
    labels[:, :4] = IGNORE
    beta = 0.1
    loss, stats = orpo_loss(logits, labels, beta)
    lp = naive_logps(logits.detach(), labels)
    c, r = lp[:2], lp[2:]
    log_odds = (c - r) - (torch.log1p(-torch.exp(c)) - torch.log1p(-torch.exp(r)))
    expected = (-c.mean()) - beta * F.logsigmoid(log_odds).mean()
    assert math.isclose(loss.item(), expected.item(), rel_tol=1e-4), (loss.item(), expected.item())
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert 0.0 <= stats["reward_acc"].item() <= 1.0


def test_collator_layout():
    feats = [{"chosen_input_ids": [1, 2, 3], "chosen_labels": [IGNORE, 2, 3],
              "rejected_input_ids": [1, 2, 3, 4, 5], "rejected_labels": [IGNORE, IGNORE, 3, 4, 5]},
             {"chosen_input_ids": [1, 9], "chosen_labels": [IGNORE, 9],
              "rejected_input_ids": [1, 8, 7], "rejected_labels": [IGNORE, 8, 7]}]
    b = PairCollator(pad_id=0)(feats)
    assert b["input_ids"].shape == (4, 5)                       # rows: chosen0, chosen1, rejected0, rejected1
    assert b["input_ids"][2].tolist() == [1, 2, 3, 4, 5]
    assert b["labels"][0].tolist() == [IGNORE, 2, 3, IGNORE, IGNORE]
    assert b["attention_mask"][1].tolist() == [1, 1, 0, 0, 0]


def test_tiny_model_learns_preference():
    """A few ORPO steps on one pair should raise the chosen-vs-rejected log-prob margin."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(2)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))
    prompt = [5, 6, 7, 8]
    feats = [{"chosen_input_ids": prompt + [10, 11, 12], "chosen_labels": [IGNORE] * 4 + [10, 11, 12],
              "rejected_input_ids": prompt + [20, 21, 22], "rejected_labels": [IGNORE] * 4 + [20, 21, 22]}]
    batch = PairCollator(pad_id=0)(feats)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def margin():
        with torch.no_grad():
            lp, _ = response_logps(model(input_ids=batch["input_ids"],
                                         attention_mask=batch["attention_mask"]).logits, batch["labels"])
        return (lp[0] - lp[1]).item()

    before = margin()
    for _ in range(15):
        loss, _ = orpo_loss(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits,
                            batch["labels"], beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    after = margin()
    assert after > before + 1.0, (before, after)


def test_trainer_wiring():
    """End-to-end on CPU: custom Trainer trains, averages eval stats over the whole val set, divides by grad-accum."""
    import tempfile
    from datasets import Dataset
    from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
    from src.train.orpo import make_trainer_class
    torch.manual_seed(3)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))
    rows = [{"chosen_input_ids": [5, 6, 7, 10 + i], "chosen_labels": [IGNORE] * 3 + [10 + i],
             "rejected_input_ids": [5, 6, 7, 30 + i, 31], "rejected_labels": [IGNORE] * 3 + [30 + i, 31]}
            for i in range(8)]
    Trainer = make_trainer_class()
    with tempfile.TemporaryDirectory() as d:
        args = TrainingArguments(output_dir=d, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                                 gradient_accumulation_steps=2, max_steps=4, learning_rate=1e-2, logging_steps=1,
                                 eval_strategy="no", save_strategy="no", report_to="none", use_cpu=True,
                                 remove_unused_columns=False)
        tr = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                     eval_dataset=Dataset.from_list(rows[:6]), data_collator=PairCollator(pad_id=0))
        assert tr.model_accepts_loss_kwargs is False
        before = tr.evaluate()
        tr.train()
        after = tr.evaluate()
    for key in ("eval_loss", "eval_reward_acc", "eval_margin", "eval_nll", "eval_or_loss"):
        assert key in after, key
    assert after["eval_margin"] > before["eval_margin"], (before["eval_margin"], after["eval_margin"])
    assert any("reward_acc" in h for h in tr.state.log_history if "loss" in h)


def test_tokenize_pair_real_tokenizer():
    import json
    from pathlib import Path
    import src.paths  # noqa: F401
    from transformers import AutoTokenizer
    from src.train.orpo import tokenize_pair
    tok = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    r = json.loads(Path("data/final/pref_prompts.jsonl").open(encoding="utf-8").readline())
    pair = {"prompt": r["messages"][:2], "chosen": [r["messages"][2]],
            "rejected": [{"role": "assistant", "content": "Calculation:\n1. 1 + 1 = 2\n\nAnswer: 2"}]}
    t = tokenize_pair(pair, tok)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    for side, text in (("chosen", r["messages"][2]["content"]), ("rejected", pair["rejected"][0]["content"])):
        ids, lab = t[f"{side}_input_ids"], t[f"{side}_labels"]
        trained = [i for i, l in zip(ids, lab) if l != IGNORE]
        assert ids.count(tok.bos_token_id) == 1 and trained[-1] == eot
        assert tok.decode(trained[:-1]) == text, (side, tok.decode(trained[:-1]))
    # identical prompt tokens on both sides
    n = t["chosen_labels"].index(next(l for l in t["chosen_labels"] if l != IGNORE))
    assert t["chosen_input_ids"][:n] == t["rejected_input_ids"][:n]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
