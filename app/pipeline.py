"""The end-to-end tool: a document in, a tagged document out.

    python -m app.cli tag tender.pdf
        1. read the .txt / .pdf and split it into clauses          (app/split.py)
        2. tag every clause: label + confidence, NEEDS REVIEW if    (app/tagger.py)
           the confidence is below the calibrated threshold
        3. write tender.tagged.md, .json and a self-contained .html (app/report.py)
"""
import json
import time
from pathlib import Path


def tag_document(file: str, adapter: str = "models/adapter.pt", threshold: float | None = None,
                 out: str | None = None) -> dict:
    from app.report import build_document, write_reports
    from app.split import document_clauses
    from app.tagger import ClauseTagger

    start = time.time()
    path = Path(file)
    clauses = document_clauses(path)
    if not clauses:
        raise SystemExit(f"no clauses found in {path}")
    tagger = ClauseTagger(adapter, threshold=threshold)
    tags = tagger.tag([c["text"] for c in clauses])
    doc = build_document(path.name, clauses, tags, tagger.labels, tagger.threshold)
    stem = Path(out) if out else path.with_name(path.stem + ".tagged")
    paths = write_reports(doc, stem)
    s = doc["summary"]
    print(f"{path.name}: {s['clauses']} clauses, {s['auto_tagged']} auto-tagged, {s['needs_review']} NEEDS REVIEW "
          f"(threshold {tagger.threshold:.2f}) in {time.time() - start:.0f}s")
    for name, n in s["auto_tagged_by_label"].items():
        if n:
            print(f"  {name:<20} {n}")
    print("wrote " + ", ".join(str(p) for p in paths))
    return doc


def real_eval(run: str = "procurement") -> dict:
    from app.real import evaluate_real
    from app.runs import get_run

    spec = get_run(run)
    m = evaluate_real(spec.adapter)
    spec.out.mkdir(parents=True, exist_ok=True)
    (spec.out / "real_eval.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"{m['clauses']} hand-labelled clauses: {m['in_set']} in the label set, {m['other']} Other "
          f"(threshold {m['threshold']:.2f})")
    print(f"  in-set accuracy (argmax)            {m['in_set_accuracy']:.1%}")
    print(f"  in-set accuracy when auto-tagged     {m['in_set_auto_accuracy']:.1%}")
    print(f"  Other clauses flagged NEEDS REVIEW   {m['other_flagged']:.1%}")
    print(f"  clerk view: auto-tags {m['auto_tagged_share']:.0%} at {m['auto_tagged_accuracy']:.1%} accuracy")
    wrong = [c for c in m["per_clause"] if c["truth"] != c["pred"] and not c["needs_review"]]
    if wrong:
        print("  confident mistakes:")
        for c in wrong[:8]:
            print(f"    truth {c['truth']:<17} said {c['pred']:<17} ({c['confidence']:.2f})  {c['text'][:70]}")
    return m
