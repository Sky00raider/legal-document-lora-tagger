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


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


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

