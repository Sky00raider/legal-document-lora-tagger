"""Standout 1: test on real Indian public records, labelled by hand.

1. `candidates`: split every PDF in data/real_docs into clauses and pick ~40 to label.
   Half are clauses whose wording hints at one of our clause types (e.g. "arbitrat",
   "terminat"); the other half are random. Without the first half almost every
   sampled clause would be procurement-specific ("Other"), and we could not measure
   in-set accuracy at all. The model is never consulted when picking, so the sample is
   not biased towards what it gets right.
2. `label_interactively`: a small terminal helper. It shows one clause at a time and
   you type its clause type, or `o` for Other (outside the label set). Answers are saved
   after every clause (app/storage.py), so you can stop and resume. The model's guess
   is never shown.
3. `evaluate_real`: tag the labelled clauses and report
     - accuracy on in-set clauses,
     - how often Other clauses are correctly flagged NEEDS REVIEW (the tagger cannot
       say "none of these"; low confidence is its only way to refuse),
     - the clerk headline: share auto-tagged and accuracy of those auto-tags
       (an auto-tagged Other clause counts as wrong).
"""
import json
import random
import re
import textwrap
from pathlib import Path

from app.storage import Store

REAL_DIR = Path("data/real_docs")
LABELS_FILE = REAL_DIR / "hand_labels.json"
OTHER = "Other"

CUES = {
    "Governing Laws": r"governed by|laws of india|governing law|applicable law",
    "Notices": r"\bnotices?\b.{0,40}\b(in writing|address|e-?mail|registered post)",
    "Amendments": r"amendment|variation to the contract|modif(y|ication) (of|to) (the|this) contract",
    "Severability": r"severab|invalid or unenforceable|held to be invalid",
    "Entire Agreements": r"entire agreement|supersedes?",
    "Terminations": r"terminat",
    "Assignments": r"\bassign(ment|ed)?\b|sub-?contract",
    "Confidentiality": r"confidential|secrecy",
    "Indemnifications": r"indemnif",
    "Taxes": r"\btax(es)?\b|\bgst\b",
    "Payments": r"payment|invoice",
    "Insurances": r"insurance",
    "Arbitration": r"arbitrat",
    "Warranties": r"warrant",
}


def all_clauses(real_dir: Path = REAL_DIR) -> list[dict]:
    from app.split import document_clauses

    out = []
    for pdf in sorted(real_dir.glob("*.pdf")):
        out += [{"doc": pdf.name, **c} for c in document_clauses(pdf)]
    return out


def candidates(clauses: list[dict], n: int = 40, seed: int = 0, max_words: int = 220) -> list[dict]:
    """About n clauses: half cue-matched (spread over clause types and documents), half random."""
    rng = random.Random(seed)
    pool = [c for c in clauses if len(c["text"].split()) <= max_words]
    chosen, used = [], set()
    cued = {name: [c for c in pool if re.search(pat, c["text"], re.I)] for name, pat in CUES.items()}
    while len(chosen) < n // 2 and any(cued.values()):
        for name in CUES:                              # round-robin over clause types
            options = [c for c in cued[name] if (c["doc"], c["id"]) not in used]
            if options and len(chosen) < n // 2:
                pick = rng.choice(options)
                used.add((pick["doc"], pick["id"]))
                chosen.append(pick)
            cued[name] = options
    rest = [c for c in pool if (c["doc"], c["id"]) not in used]
    chosen += rng.sample(rest, min(n - len(chosen), len(rest)))
    rng.shuffle(chosen)
    return [{"key": f"{c['doc']}#{c['id']}", "doc": c["doc"], "number": c["number"], "text": c["text"]}
            for c in chosen]


def label_interactively(labels: list[str], n: int = 40, path: Path = LABELS_FILE, ask=input, show=print) -> None:
    """Terminal helper: label each candidate clause, saved after every answer, resumable."""
    store = Store(str(path))
    queue = store.get("_queue")
    if not queue:
        show("Splitting the PDFs in data/real_docs into clauses ...")
        queue = candidates(all_clauses(), n)
        store.put("_queue", queue)
        store.put("_label_set", labels)
    menu = "  ".join(f"{i + 1}={name}" for i, name in enumerate(labels))
    i = 0
    while i < len(queue):
        item = queue[i]
        if store.get(item["key"]):
            i += 1
            continue
        done = sum(bool(store.get(q["key"])) for q in queue)
        show(f"\n[{done + 1}/{len(queue)}] {item['doc']}  clause {item['number'] or '-'}")
        show(textwrap.fill(item["text"], 100))
        show("\n" + textwrap.fill(menu, 100))
        answer = ask(f"type 1-{len(labels)}, o=Other (none of these), s=skip, b=back, q=quit > ").strip().lower()
        if answer == "q":
            break
        if answer == "b":
            prev = [q for q in queue[:i] if store.get(q["key"])]
            if prev:
                store.put(prev[-1]["key"], None)
                i = queue.index(prev[-1])
            continue
        if answer == "s":
            store.put(item["key"], {"label": None, "skipped": True, **item})
        elif answer == "o":
            store.put(item["key"], {"label": OTHER, "source": "hand", **item})
        elif answer.isdigit() and 1 <= int(answer) <= len(labels):
            store.put(item["key"], {"label": labels[int(answer) - 1], "source": "hand", **item})
        else:
            show("?? not understood")
            continue
        i += 1
    labelled = load_hand_labels(path)
    show(f"\n{len(labelled)} clauses labelled ({sum(r['label'] == OTHER for r in labelled)} Other). Saved to {path}")


def review_labels(labels: list[str], path: Path = LABELS_FILE, ask=input, show=print) -> None:
    """Go through draft labels: Enter keeps one, a number or o changes it, q stops. Reviewed ones become 'hand'."""
    store = Store(str(path))
    drafts = [r for r in load_hand_labels(path) if r.get("source") == "draft"]
    menu = "  ".join(f"{i + 1}={name}" for i, name in enumerate(labels))
    for n, row in enumerate(drafts):
        show(f"\n[{n + 1}/{len(drafts)}] {row['doc']}  clause {row['number'] or '-'}")
        show(textwrap.fill(row["text"], 100))
        show(f"\ndraft label: {row['label']}" + (f"   (note: {row['note']})" if row.get("note") else ""))
        show(textwrap.fill(menu, 100))
        answer = ask("Enter = keep, 1-%d or o = change, q = quit > " % len(labels)).strip().lower()
        if answer == "q":
            break
        if answer == "o":
            row["label"] = OTHER
        elif answer.isdigit() and 1 <= int(answer) <= len(labels):
            row["label"] = labels[int(answer) - 1]
        elif answer:
            show("?? not understood, kept as draft")
            continue
        row["source"] = "hand"
        store.put(row["key"], row)
    left = sum(r.get("source") == "draft" for r in load_hand_labels(path))
    show(f"\n{left} draft labels left unreviewed.")


def load_hand_labels(path: Path = LABELS_FILE) -> list[dict]:
    if not Path(path).exists():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [v for k, v in data.items() if not k.startswith("_") and v and v.get("label")]


def real_metrics(rows: list[dict], tags: list[dict], labels: list[str]) -> dict:
    """Accuracy on in-set clauses, NEEDS REVIEW rate on Other clauses, and the clerk headline."""
    in_set = [(r, t) for r, t in zip(rows, tags) if r["label"] in labels]
    other = [(r, t) for r, t in zip(rows, tags) if r["label"] not in labels]
    auto = [(r, t) for r, t in zip(rows, tags) if not t["needs_review"]]
    in_auto = [(r, t) for r, t in in_set if not t["needs_review"]]
    rate = lambda xs, f: (sum(f(r, t) for r, t in xs) / len(xs)) if xs else float("nan")
    return {
        "clauses": len(rows), "in_set": len(in_set), "other": len(other),
        "in_set_accuracy": rate(in_set, lambda r, t: t["label"] == r["label"]),
        "in_set_auto_accuracy": rate(in_auto, lambda r, t: t["label"] == r["label"]),
        "in_set_flagged": rate(in_set, lambda r, t: t["needs_review"]),
        "other_flagged": rate(other, lambda r, t: t["needs_review"]),
        "auto_tagged_share": len(auto) / len(rows) if rows else float("nan"),
        "auto_tagged_accuracy": rate(auto, lambda r, t: t["label"] == r["label"]),
    }


def evaluate_real(adapter: Path, threshold: float | None = None, path: Path = LABELS_FILE) -> dict:
    from app.tagger import ClauseTagger

    rows = load_hand_labels(path)
    if not rows:
        raise SystemExit(f"no hand labels yet: run  python -m app.cli label  first ({path})")
    tagger = ClauseTagger(adapter, threshold=threshold)
    tags = tagger.tag([r["text"] for r in rows])
    m = real_metrics(rows, tags, tagger.labels)
    m["threshold"] = tagger.threshold
    m["per_clause"] = [{"key": r["key"], "truth": r["label"], "pred": t["label"], "confidence": t["confidence"],
                        "needs_review": t["needs_review"], "text": r["text"][:200]} for r, t in zip(rows, tags)]
    return m
