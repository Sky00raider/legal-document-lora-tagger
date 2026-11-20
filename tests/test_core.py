"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from app import data


# Milestone 1: Prepare a dataset of legal clauses for tagging
def _rows(spec: dict[str, int]) -> list[dict]:
    return [{"text": f"{label} clause number {i}.", "label": label}
            for label, n in spec.items() for i in range(n)]


def test_milestone_1_example_format_ends_with_one_label_token():
    assert data.label_token("Governing Laws") == "<governing_laws>"
    assert data.format_example("Governed by\n  New York   law.", "Governing Laws") == \
        "Governed by New York law. <governing_laws>"


def test_milestone_1_top_labels_and_balanced_sample():
    rows = _rows({"Notices": 30, "Waivers": 20, "Severability": 5})
    assert data.top_labels(rows, 2) == ["Notices", "Waivers"]
    sample = data.balanced_sample(rows, ["Notices", "Waivers"], per_class=10, seed=1)
    labels = [r["label"] for r in sample]
    assert labels.count("Notices") == 10 and labels.count("Waivers") == 10
    assert sample == data.balanced_sample(rows, ["Notices", "Waivers"], per_class=10, seed=1)


def test_milestone_1_dedupe_and_no_train_test_overlap():
    rows = [{"text": "Same  Clause.", "label": "A"}, {"text": "same clause.", "label": "B"},
            {"text": "Other.", "label": "A"}]
    assert [r["label"] for r in data.dedupe(rows)] == ["A", "A"]
    assert data.drop_overlap([{"text": "SAME clause.", "label": "A"}, {"text": "new", "label": "A"}],
                             rows) == [{"text": "new", "label": "A"}]


def test_milestone_1_build_dataset_on_fake_splits():
    splits = {"train": _rows({"A": 40, "B": 30, "C": 3}),
              "validation": _rows({"A": 2}) + [{"text": f"val {i}", "label": "B"} for i in range(9)],
              "test": [{"text": f"test {i}", "label": "A"} for i in range(9)]}
    ds = data.build_dataset(splits, top_k=2, per_class=10, eval_per_class=5)
    assert ds["labels"] == ["A", "B"] and len(ds["train"]) == 20
    assert all(r["text"].startswith("val") for r in ds["val"])   # copies of train text removed
    assert len(ds["test"]) == 5


# Shared tiny model: a 2-layer GPT-2 built from a config, random weights, no download.
def tiny_gpt2(vocab: int = 50, n_labels: int = 3):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_positions=32, n_embd=16, n_layer=2, n_head=2))
    model.resize_token_embeddings(vocab + n_labels)
    return model.eval(), list(range(vocab, vocab + n_labels))


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2_only_label_rows_get_gradient():
    from app.model import train_only_label_rows
    model, label_ids = tiny_gpt2()
    for p in model.parameters():
        p.requires_grad_(False)
    train_only_label_rows(model, label_ids)
    model(torch.tensor([[1, 2, 3, label_ids[0]]]), labels=torch.tensor([[1, 2, 3, label_ids[0]]])).loss.backward()
    grad = model.get_input_embeddings().weight.grad
    assert grad[label_ids].abs().sum() > 0
    assert grad[:label_ids[0]].abs().sum() == 0          # the original vocabulary never moves


def test_milestone_2_label_logits_match_full_forward():
    from app.model import label_logits
    model, label_ids = tiny_gpt2()
    prompts = [[1, 2, 3], [4, 5]]
    scores = label_logits(model, prompts, label_ids, pad_id=0)
    for i, p in enumerate(prompts):
        full = model(torch.tensor([p])).logits[0, -1]
        assert torch.allclose(scores[i], full[label_ids], atol=1e-5)


@pytest.mark.slow
def test_milestone_2_real_tokenizer_label_token_is_atomic():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.model import ClauseEncoder, add_label_tokens
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    ids = add_label_tokens(model, tok, ["Governing Laws", "Notices"])
    assert tok.tokenize("<governing_laws>") == ["<governing_laws>"] and tok.pad_token == tok.eos_token
    text = "This Agreement is governed by Delaware law."
    expected = tok(data.format_example(text, "Notices"), add_special_tokens=False).input_ids
    assert ClauseEncoder(tok, ids).training_examples([text], [1])[0] == expected


# Milestone 3: Implement LoRA adapter and inject into GPT-2
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

