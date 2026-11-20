"""Tests for the extras beyond the six milestones: real-record labelling and evaluation,
clerk triage / calibration, the tag pipeline. Fast ones run in CI; `slow` ones need the
base model download and the committed adapter (run locally with `pytest -m slow`)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from app import real
from app.data import PROCUREMENT_LABELS, build_dataset
from app.runs import RUNS, get_run


def test_runs_have_separate_paths_and_valid_labels():
    assert get_run("top6").top_k == 6 and get_run("procurement").labels == tuple(PROCUREMENT_LABELS)
    assert len({r.adapter for r in RUNS.values()}) == len(RUNS)
    assert get_run("procurement").adapter == Path("models/adapter.pt")       # the shipped one
    with pytest.raises(ValueError):
        get_run("nope")


def test_explicit_label_list_must_exist_in_ledgar():
    rows = [{"text": f"t{i}", "label": "A"} for i in range(5)]
    splits = {"train": rows, "validation": [], "test": []}
    assert build_dataset(splits, 6, 2, 1, labels=["A"])["labels"] == ["A"]
    with pytest.raises(ValueError):
        build_dataset(splits, 6, 2, 1, labels=["A", "Force Majeure"])


def _clauses():
    texts = ["Disputes shall be referred to arbitration under the Arbitration Act.",
             "The contract may be terminated by the purchaser for default.",
             "GST shall be paid as applicable at the time of supply.",
             "The bidder shall submit samples within ten days of the notice."] * 5
    return [{"doc": f"d{i % 2}.pdf", "id": i, "number": str(i), "text": t} for i, t in enumerate(texts)]


def test_candidates_mix_cued_and_random_and_are_reproducible():
    picked = real.candidates(_clauses(), n=8, seed=3)
    assert len(picked) == 8 and len({p["key"] for p in picked}) == 8
    assert picked == real.candidates(_clauses(), n=8, seed=3)
    assert any("arbitration" in p["text"] for p in picked) and any("terminated" in p["text"] for p in picked)


def test_labelling_helper_saves_resumes_and_goes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(real, "all_clauses", _clauses)
    path = tmp_path / "labels.json"
    answers = iter(["1", "o", "b", "2", "zzz", "s", "q"])
    real.label_interactively(["Arbitration", "Terminations"], n=4, path=path, ask=lambda _: next(answers),
                             show=lambda *_: None)
    rows = real.load_hand_labels(path)
    assert [r["label"] for r in rows] == ["Arbitration", "Terminations"]    # "o" was undone by "b"
    answers = iter(["o", "q"])                                               # resume: next unlabelled clause
    real.label_interactively(["Arbitration", "Terminations"], n=4, path=path, ask=lambda _: next(answers),
                             show=lambda *_: None)
    assert [r["label"] for r in real.load_hand_labels(path)] == ["Arbitration", "Terminations", "Other"]


def test_review_mode_keeps_or_changes_draft_labels(tmp_path):
    path = tmp_path / "labels.json"
    rows = {f"d#{i}": {"key": f"d#{i}", "doc": "d", "number": str(i), "text": f"clause {i}",
                       "label": "Taxes", "source": "draft"} for i in range(3)}
    path.write_text(json.dumps(rows), encoding="utf-8")
    answers = iter(["", "o", "q"])                       # keep, change to Other, stop
    real.review_labels(["Taxes", "Payments"], path=path, ask=lambda _: next(answers), show=lambda *_: None)
    got = {r["key"]: (r["label"], r["source"]) for r in real.load_hand_labels(path)}
    assert got == {"d#0": ("Taxes", "hand"), "d#1": ("Other", "hand"), "d#2": ("Taxes", "draft")}


def test_real_metrics_counts_other_and_review_correctly():
    rows = [{"label": "Taxes"}, {"label": "Taxes"}, {"label": "Other"}, {"label": "Other"}]
    tags = [{"label": "Taxes", "needs_review": False}, {"label": "Payments", "needs_review": True},
            {"label": "Payments", "needs_review": True}, {"label": "Taxes", "needs_review": False}]
    m = real.real_metrics(rows, tags, ["Taxes", "Payments"])
    assert m["in_set"] == 2 and m["other"] == 2
    assert m["in_set_accuracy"] == 0.5 and m["in_set_auto_accuracy"] == 1.0
    assert m["other_flagged"] == 0.5
    assert m["auto_tagged_share"] == 0.5 and m["auto_tagged_accuracy"] == 0.5   # auto-tagged Other is wrong


def test_threshold_choice_and_saving_into_the_adapter(tmp_path):
    from app.calibrate import conf_correct, save_threshold
    from app.metrics import at_threshold, threshold_for_accuracy
    logits = np.array([[4.0, 0, 0], [3.0, 0, 0], [0.5, 0, 0], [0.2, 0.1, 0]])
    target = np.array([0, 0, 1, 0])
    conf, ok = conf_correct(logits, target)
    t = threshold_for_accuracy(conf, ok, 0.99)
    assert at_threshold(conf, ok, t)["accuracy"] == 1.0 and at_threshold(conf, ok, t)["coverage"] == 0.5
    path = tmp_path / "a.pt"
    torch.save({"config": {"extra": {}}, "lora": {}}, path)
    save_threshold(path, t, 0.99)
    assert torch.load(path, weights_only=True)["config"]["extra"]["threshold"] == pytest.approx(t)


def test_plots_are_written(tmp_path):
    from app.calibrate import plot_coverage, plot_reliability
    from app.metrics import reliability
    conf, ok = np.array([0.9, 0.8, 0.6, 0.4]), np.array([1, 1, 0, 1])
    plot_coverage({"a": (np.array([0.25, 0.5, 1.0]), np.array([1.0, 1.0, 0.75]))}, {"a": (0.5, 1.0)}, tmp_path / "c.png")
    plot_reliability(reliability(conf, ok, bins=5), tmp_path / "r.png", "t")
    assert (tmp_path / "c.png").stat().st_size > 1000 and (tmp_path / "r.png").stat().st_size > 1000


@pytest.mark.slow
def test_tag_pipeline_end_to_end_with_committed_adapter(tmp_path):
    from app.pipeline import tag_document
    if not Path("models/adapter.pt").exists():
        pytest.skip("no trained adapter")
    src = tmp_path / "doc.txt"
    src.write_text("1. This Agreement shall be governed by the laws of India and the courts at Delhi.\n"
                   "2. Any dispute shall be settled by arbitration under the Arbitration and Conciliation Act.\n",
                   encoding="utf-8")
    doc = tag_document(str(src))
    assert doc["summary"]["clauses"] == 2
    assert json.loads((tmp_path / "doc.tagged.json").read_text(encoding="utf-8"))["clauses"][0]["label"]
    assert (tmp_path / "doc.tagged.html").exists() and (tmp_path / "doc.tagged.md").exists()
