"""Milestone 6: how well does the tagger work, and what does LoRA cost?

Accuracy:   test split (50 clauses per type, none seen verbatim in training):
            accuracy, macro-F1, per-class precision / recall / F1, most confused pairs.
Efficiency: trainable parameters, adapter file size vs the size of a full fine-tuned
            checkpoint, and training time (from <run outputs>/train_report.json).

The val and test label scores are saved to <run outputs>/ledgar_logits.npz, so the
calibration / threshold analysis (standout 2) runs without the model.
"""
import json
from pathlib import Path

import numpy as np

from app.data import load_split
from app.metrics import confused_pairs, macro_f1, per_class, softmax
from app.runs import get_run
from app.tagger import ClauseTagger


def ledgar_scores(tagger: ClauseTagger, data_dir: Path) -> dict[str, np.ndarray]:
    out = {}
    for split in ("val", "test"):
        rows = load_split(split, data_dir)
        out[f"{split}_logits"] = tagger.scores([r["text"] for r in rows])
        out[f"{split}_target"] = np.array([tagger.labels.index(r["label"]) for r in rows])
    return out


def results_markdown(labels, stats, acc, f1, pairs, eff) -> str:
    lines = ["## LEDGAR test set", "",
             f"**Accuracy {acc:.1%}, macro-F1 {f1:.1%}** on {int(stats['support'].sum())} held-out clauses "
             f"({len(labels)} clause types, {int(stats['support'][0])} each).", "",
             "| Clause type | Precision | Recall | F1 | Test clauses |", "|---|---:|---:|---:|---:|"]
    for i, name in enumerate(labels):
        lines.append(f"| {name} | {stats['precision'][i]:.2f} | {stats['recall'][i]:.2f} | "
                     f"{stats['f1'][i]:.2f} | {stats['support'][i]} |")
    lines += ["", "Most confused pairs (true -> predicted, both directions):", ""]
    lines += [f"- {labels[a]} <-> {labels[b]}: {ab} + {ba}" for a, b, ab, ba in pairs] or ["- none"]
    lines += ["", "## Efficiency", "", "| | |", "|---|---:|",
              f"| Parameters in distilgpt2 | {eff['total_params']:,} |",
              f"| Trainable (LoRA A/B + {len(labels)} label rows) | {eff['trainable_params']:,} "
              f"({100 * eff['trainable_params'] / eff['total_params']:.2f}%) |",
              f"| LoRA A/B only | {eff['lora_params']:,} ({100 * eff['lora_params'] / eff['total_params']:.2f}%) |",
              f"| Adapter file (what we ship) | {eff['adapter_kb']:.0f} KB |",
              f"| Full fine-tuned checkpoint (float32) | {eff['full_model_mb']:.0f} MB |",
              f"| Adapter / full model | 1 : {eff['full_model_mb'] * 1024 / eff['adapter_kb']:.0f} |",
              f"| Training time (laptop CPU, {eff['train_examples']} clauses x {eff['epochs']:g} epochs) | "
              f"{eff['train_seconds'] / 60:.1f} min |"]
    return "\n".join(lines) + "\n"


def evaluate_ledgar(run: str = "top6") -> dict:
    spec = get_run(run)
    OUT, adapter_path = spec.out, spec.adapter
    tagger = ClauseTagger(adapter_path)
    labels, k = tagger.labels, len(tagger.labels)
    scores = ledgar_scores(tagger, spec.data_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "ledgar_logits.npz", labels=np.array(labels), **scores)

    target = scores["test_target"]
    pred = softmax(scores["test_logits"]).argmax(axis=1)
    acc = float((pred == target).mean())
    f1 = macro_f1(pred, target, k)
    stats = per_class(pred, target, k)
    pairs = confused_pairs(pred, target, top=5)

    train = json.loads((OUT / "train_report.json").read_text(encoding="utf-8"))
    base_params = train["total_params"] - k * tagger.model.config.n_embd       # distilgpt2 without our rows
    eff = {"total_params": train["total_params"], "trainable_params": train["trainable_params"],
           "lora_params": train["lora_params"], "adapter_kb": Path(adapter_path).stat().st_size / 1024,
           "full_model_mb": base_params * 4 / 1024 ** 2, "train_seconds": train["train_seconds"],
           "train_examples": train["train_examples"], "epochs": train["epochs"]}

    print(f"test accuracy {acc:.3f}, macro-F1 {f1:.3f}  ({len(target)} clauses)")
    print(f"{'clause type':<22} {'prec':>5} {'rec':>5} {'f1':>5}")
    for i, name in enumerate(labels):
        print(f"{name:<22} {stats['precision'][i]:5.2f} {stats['recall'][i]:5.2f} {stats['f1'][i]:5.2f}")
    print("confusions:", ", ".join(f"{labels[a]}<->{labels[b]} {ab}+{ba}" for a, b, ab, ba in pairs) or "none")
    print(f"trainable {eff['trainable_params']:,} of {eff['total_params']:,} "
          f"({100 * eff['trainable_params'] / eff['total_params']:.2f}%), adapter {eff['adapter_kb']:.0f} KB vs "
          f"full model {eff['full_model_mb']:.0f} MB, trained in {eff['train_seconds'] / 60:.1f} min")

    result = {"accuracy": acc, "macro_f1": f1, "per_class": {
        name: {m: float(stats[m][i]) for m in ("precision", "recall", "f1")} for i, name in enumerate(labels)},
        "confusions": [[labels[a], labels[b], ab, ba] for a, b, ab, ba in pairs], "efficiency": eff}
    (OUT / "eval_ledgar.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "results_ledgar.md").write_text(results_markdown(labels, stats, acc, f1, pairs, eff), encoding="utf-8")
    return result
