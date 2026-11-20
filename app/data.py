"""Milestone 1: a balanced dataset of legal clauses, each written as "clause text <Label>".

Source: LEDGAR (coastalcph/lex_glue, config "ledgar"): ~80,000 clauses from US SEC
contracts, each labelled with one of 100 clause types (Governing Laws, Notices, ...).

We do not train on all 100 types. We pick the `top_k` most common types in LEDGAR's
train split and draw `per_class` examples of each, so every class is equally
represented and training fits on a laptop CPU. The label set is configurable:
`python -m app.cli prepare --top-k 12 --per-class 100`.

Each example becomes one string the language model learns to complete:

    "This Agreement shall be governed by the laws of the State of New York. <governing_laws>"

where <governing_laws> is a single new token (added in milestone 2), so the model
answers with exactly one token per clause.

Contracts reuse boilerplate, so the same clause text appears many times. Exact repeats
(after lower-casing and collapsing whitespace) are dropped before sampling, and any
validation/test clause whose text also occurs in train is removed, so evaluation
never scores a clause the model saw verbatim.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data/ledgar")


def label_token(name: str) -> str:
    """'Governing Laws' -> '<governing_laws>': one atomic token per clause type."""
    return "<" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ">"


def format_example(text: str, label: str) -> str:
    """'clause text <Label>' : the exact string the model is trained on."""
    return f"{' '.join(text.split())} {label_token(label)}"


def normalize(text: str) -> str:
    """Key for duplicate detection: lower-case, whitespace collapsed."""
    return re.sub(r"\s+", " ", text).strip().lower()


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep the first occurrence of each normalised clause text."""
    seen, kept = set(), []
    for row in rows:
        key = normalize(row["text"])
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept


def top_labels(rows: list[dict], k: int) -> list[str]:
    """The k most frequent labels, most frequent first (ties broken by name)."""
    counts = Counter(row["label"] for row in rows)
    return [name for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def balanced_sample(rows: list[dict], labels: list[str], per_class: int, seed: int = 0) -> list[dict]:
    """Up to `per_class` rows of each label in `labels`, shuffled. Same seed, same sample."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for row in rows:
        if row["label"] in labels:
            by_label[row["label"]].append(row)
    out = []
    for name in labels:
        members = by_label[name]
        out.extend(rng.sample(members, min(per_class, len(members))))
    rng.shuffle(out)
    return out


def drop_overlap(rows: list[dict], reference: list[dict]) -> list[dict]:
    """Rows whose normalised text does not occur in `reference` (no train/test leakage)."""
    seen = {normalize(r["text"]) for r in reference}
    return [r for r in rows if normalize(r["text"]) not in seen]


def load_ledgar() -> dict[str, list[dict]]:
    """LEDGAR's three splits as [{"text", "label"}] with the label as its name (downloads once)."""
    from datasets import load_dataset   # imported here so tests and CI never need it

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    names = ds["train"].features["label"].names
    return {split: [{"text": r["text"], "label": names[r["label"]]} for r in ds[split]]
            for split in ("train", "validation", "test")}


def build_dataset(splits: dict[str, list[dict]], top_k: int, per_class: int,
                  eval_per_class: int, seed: int = 0) -> dict:
    """Pick the label set and the balanced train / val / test samples from LEDGAR splits."""
    train_all = dedupe(splits["train"])
    labels = top_labels(train_all, top_k)
    train = balanced_sample(train_all, labels, per_class, seed)
    out = {"labels": labels, "train": train}
    for split, name in (("validation", "val"), ("test", "test")):
        clean = drop_overlap(dedupe(splits[split]), train_all)
        out[name] = balanced_sample(clean, labels, eval_per_class, seed)
    out["label_counts_in_ledgar_train"] = {name: sum(r["label"] == name for r in train_all) for name in labels}
    return out


def save_dataset(dataset: dict, data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "labels.json").write_text(json.dumps(dataset["labels"], indent=2), encoding="utf-8")
    for split in ("train", "val", "test"):
        with (data_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in dataset[split]:
                f.write(json.dumps({**row, "example": format_example(row["text"], row["label"])}) + "\n")


def load_labels(data_dir: Path = DATA_DIR) -> list[str]:
    return json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    with (data_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
