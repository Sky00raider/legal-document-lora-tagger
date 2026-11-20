"""Standout 2: clerk triage. When should the tagger answer, and when should a clerk look?

The tagger auto-tags a clause only when its confidence (softmax over the label tokens)
is at least a threshold; everything below goes to a clerk as NEEDS REVIEW. Raising the
threshold lowers coverage (share auto-tagged) and, if confidence means anything, raises
the accuracy of what is auto-tagged.

  * The threshold is chosen on LEDGAR *validation*: the lowest one whose auto-tagged
    clauses reach the target accuracy (default 98%: a clerk tool should be conservative,
    and with 95% the threshold fell to 0.42 and flagged almost nothing on real records). It is then applied unchanged to
    LEDGAR test and to the real documents, and saved inside the adapter so `tag` uses it.
  * Reliability diagram: among predictions made with confidence about p, is a share of
    about p correct? (ECE = the average gap, weighted by how many predictions per bin.)
  * On the real documents, an auto-tagged "Other" clause counts as an error: the
    tagger should have asked for help.
"""
import json
from pathlib import Path

import numpy as np
import torch

from app.metrics import at_threshold, coverage_curve, reliability, softmax, threshold_for_accuracy
from app.runs import get_run

INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 of the reference palette


def conf_correct(logits: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = softmax(logits)
    return p.max(axis=1), (p.argmax(axis=1) == target)


def real_conf_correct(real_rows: list[dict], scores: np.ndarray, labels: list[str]):
    """Confidence and correctness on labelled real clauses (Other is never correct)."""
    p = softmax(scores)
    truth = [r["label"] for r in real_rows]
    correct = np.array([labels[i] == t for i, t in zip(p.argmax(axis=1), truth)])
    return p.max(axis=1), correct


def threshold_table(conf, correct, thresholds) -> list[dict]:
    return [at_threshold(conf, correct, t) for t in thresholds]


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")


def plot_coverage(curves: dict, picks: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax, "Accuracy of auto-tagged clauses vs share auto-tagged")
    for (name, (cov, acc)), color in zip(curves.items(), (BLUE, ORANGE)):
        ax.plot(cov * 100, acc * 100, color=color, linewidth=2, label=name)
        pc, pa = picks[name]
        ax.plot(pc * 100, pa * 100, "o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{pc:.0%} auto-tagged at {pa:.0%}", (pc * 100, pa * 100), xytext=(8, -14),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlabel("clauses auto-tagged (%)  - the rest go to a clerk", color=INK_2)
    ax.set_ylabel("accuracy of auto-tags (%)", color=INK_2)
    ax.set_xlim(0, 101)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(rel: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2), facecolor=SURFACE)
    _style(ax, title)
    edges, count = rel["edges"], rel["count"]
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    keep = count > 0
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="perfect calibration")
    ax.bar(centers[keep], rel["accuracy"][keep], width=width * 0.9, color=BLUE, label="actual accuracy")
    for c, a, n in zip(centers[keep], rel["accuracy"][keep], count[keep]):
        ax.text(c, a + 0.02, str(n), ha="center", color=INK_2, fontsize=8)
    ax.set_xlabel("confidence (numbers = clauses in bin)", color=INK_2)
    ax.set_ylabel("share correct", color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    ax.text(0.98, 0.04, f"ECE {rel['ece']:.3f}", ha="right", color=INK, fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def save_threshold(adapter: Path, threshold: float, target: float) -> None:
    """Store the chosen threshold inside the adapter file (the tensors are untouched)."""
    saved = torch.load(adapter, map_location="cpu", weights_only=True)
    saved["config"]["extra"]["threshold"] = float(threshold)
    saved["config"]["extra"]["threshold_target_accuracy"] = target
    torch.save(saved, adapter)


def calibrate(run: str, target: float = 0.98) -> dict:
    from app.real import load_hand_labels

    spec = get_run(run)
    npz = np.load(spec.out / "ledgar_logits.npz")
    labels = [str(x) for x in npz["labels"]]
    val_conf, val_ok = conf_correct(npz["val_logits"], npz["val_target"])
    test_conf, test_ok = conf_correct(npz["test_logits"], npz["test_target"])
    threshold = threshold_for_accuracy(val_conf, val_ok, target)

    result = {"target_accuracy": target, "threshold": threshold,
              "ledgar_val": at_threshold(val_conf, val_ok, threshold),
              "ledgar_test": at_threshold(test_conf, test_ok, threshold),
              "ledgar_test_reliability_ece": reliability(test_conf, test_ok)["ece"],
              "ledgar_test_table": threshold_table(test_conf, test_ok, [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, threshold])}
    cov, acc, _ = coverage_curve(test_conf, test_ok)
    curves = {"LEDGAR test (US contracts)": (cov, acc)}
    picks = {"LEDGAR test (US contracts)": (result["ledgar_test"]["coverage"], result["ledgar_test"]["accuracy"])}

    rows = load_hand_labels()
    if rows:
        from app.tagger import ClauseTagger
        tagger = ClauseTagger(spec.adapter)
        r_conf, r_ok = real_conf_correct(rows, tagger.scores([r["text"] for r in rows]), labels)
        result["real"] = at_threshold(r_conf, r_ok, threshold)
        result["real_table"] = threshold_table(r_conf, r_ok, [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, threshold])
        cov, acc, _ = coverage_curve(r_conf, r_ok)
        curves["Indian public records (40 labelled clauses)"] = (cov, acc)
        picks["Indian public records (40 labelled clauses)"] = (result["real"]["coverage"], result["real"]["accuracy"])
        plot_reliability(reliability(r_conf, r_ok, bins=5), spec.out / "reliability_real.png",
                         "Reliability: Indian public records")

    plot_coverage(curves, picks, spec.out / "coverage.png")
    plot_reliability(reliability(test_conf, test_ok), spec.out / "reliability_ledgar.png", "Reliability: LEDGAR test")
    save_threshold(spec.adapter, threshold, target)
    (spec.out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    t = result["ledgar_test"]
    print(f"threshold {threshold:.4f} (lowest reaching {target:.0%} accuracy on LEDGAR validation), saved in {spec.adapter}")
    print(f"LEDGAR test: auto-tags {t['coverage']:.0%} of clauses at {t['accuracy']:.1%} accuracy, "
          f"sends {1 - t['coverage']:.0%} to a clerk (ECE {result['ledgar_test_reliability_ece']:.3f})")
    if "real" in result:
        t = result["real"]
        print(f"Real records: auto-tags {t['coverage']:.0%} of clauses at {t['accuracy']:.1%} accuracy, "
              f"sends {1 - t['coverage']:.0%} to a clerk")
    return result
