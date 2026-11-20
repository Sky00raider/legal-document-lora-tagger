"""The tagged document, written three ways: JSON (for software), Markdown (for GitHub),
and one self-contained HTML file (for a clerk: no internet, no scripts, opens anywhere).

The HTML report shows summary counts at the top, then every clause in document order,
colour-coded by its label, with a confidence bar. Clauses below the review threshold
are highlighted NEEDS REVIEW and also listed first in a "to review" section.
"""
import html
import json
from collections import Counter
from datetime import date
from pathlib import Path

# Distinct, colour-blind-friendlier hues; label i gets PALETTE[i % len].
PALETTE = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#0891b2",
           "#65a30d", "#dc2626", "#4f46e5", "#0d9488", "#ca8a04", "#9333ea",
           "#e11d48", "#0284c7", "#16a34a"]


def summarize(tagged: list[dict], labels: list[str]) -> dict:
    review = sum(c["needs_review"] for c in tagged)
    auto = [c for c in tagged if not c["needs_review"]]
    counts = Counter(c["label"] for c in auto)
    return {"clauses": len(tagged), "auto_tagged": len(auto), "needs_review": review,
            "auto_tagged_by_label": {name: counts.get(name, 0) for name in labels}}


def to_json(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False)


def to_markdown(doc: dict) -> str:
    s = doc["summary"]
    lines = [f"# Tagged document: {doc['source']}", "",
             f"{s['clauses']} clauses: **{s['auto_tagged']} auto-tagged**, "
             f"**{s['needs_review']} NEEDS REVIEW** (confidence below {doc['threshold']:.2%}).", "",
             "| Clause type | Auto-tagged |", "|---|---:|"]
    lines += [f"| {name} | {n} |" for name, n in s["auto_tagged_by_label"].items()]
    lines += ["", "| # | Clause | Tag | Confidence |", "|---|---|---|---:|"]
    for c in doc["clauses"]:
        text = c["text"].replace("|", "\\|")
        text = text if len(text) <= 160 else text[:157] + "..."
        tag = f"**NEEDS REVIEW** (best guess: {c['label']})" if c["needs_review"] else c["label"]
        lines.append(f"| {c['number'] or c['id']} | {text} | {tag} | {c['confidence']:.1%} |")
    return "\n".join(lines) + "\n"


CSS = """
:root{--bg:#f8fafc;--card:#fff;--text:#0f172a;--muted:#64748b;--line:#e2e8f0;--review:#fef3c7;--review-line:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#0b1120;--card:#111827;--text:#e5e7eb;--muted:#94a3b8;--line:#1f2937;--review:#3b2a06;--review-line:#f59e0b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 12px}
.muted{color:var(--muted)}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.stat b{display:block;font-size:24px}
.legend{display:flex;flex-wrap:wrap;gap:8px}.chip{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:2px 10px;font-size:13px;color:#fff;white-space:nowrap}
.clause{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--c);border-radius:10px;padding:12px 14px;margin:10px 0}
.clause.review{background:var(--review);border-left-color:var(--review-line);border-left-style:dashed}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:6px}.num{font-weight:600;color:var(--muted);min-width:40px}
.flag{font-weight:700;color:var(--review-line);font-size:13px;letter-spacing:.03em}
.bar{flex:0 0 120px;height:8px;border-radius:4px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%;background:var(--c)}
.review .bar i{background:var(--review-line)}.conf{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted)}
.text{white-space:pre-wrap;word-wrap:break-word}table{border-collapse:collapse;width:100%}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}
a{color:inherit}
"""


def _clause_html(c: dict, colors: dict) -> str:
    color = colors.get(c["label"], "#64748b")
    pct = 100 * c["confidence"]
    flag = (f'<span class="flag">NEEDS REVIEW</span><span class="muted">best guess</span>' if c["needs_review"] else "")
    return (f'<div class="clause{" review" if c["needs_review"] else ""}" id="c{c["id"]}" style="--c:{color}">'
            f'<div class="head"><span class="num">{html.escape(str(c["number"] or c["id"]))}</span>{flag}'
            f'<span class="chip" style="background:{color}">{html.escape(c["label"])}</span>'
            f'<span class="bar" title="confidence {pct:.1f}%"><i style="width:{pct:.1f}%"></i></span>'
            f'<span class="conf">{pct:.1f}%</span></div>'
            f'<div class="text">{html.escape(c["text"])}</div></div>')


def to_html(doc: dict) -> str:
    s, labels = doc["summary"], doc["labels"]
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(labels)}
    share = 100 * s["auto_tagged"] / max(s["clauses"], 1)
    legend = "".join(f'<span class="chip" style="background:{colors[n]}">{html.escape(n)} · {s["auto_tagged_by_label"][n]}</span>'
                     for n in labels)
    review = [c for c in doc["clauses"] if c["needs_review"]]
    review_list = ("".join(f'<tr><td><a href="#c{c["id"]}">{html.escape(str(c["number"] or c["id"]))}</a></td>'
                           f'<td>{html.escape(c["text"][:110])}{"..." if len(c["text"]) > 110 else ""}</td>'
                           f'<td>{html.escape(c["label"])}?</td><td>{100 * c["confidence"]:.1f}%</td></tr>'
                           for c in review) or '<tr><td colspan="4">Nothing to review.</td></tr>')
    body = "".join(_clause_html(c, colors) for c in doc["clauses"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tagged: {html.escape(doc['source'])}</title><style>{CSS}</style></head>
<body><main>
<h1>Tagged document: {html.escape(doc['source'])}</h1>
<div class="muted">Tagged {html.escape(doc['date'])} by a LoRA fine-tuned distilgpt2. Clauses with confidence below
{doc['threshold']:.2%} are marked NEEDS REVIEW: the tag is only a guess and a clerk should decide.</div>
<div class="stats">
<div class="stat"><b>{s['clauses']}</b>clauses found</div>
<div class="stat"><b>{s['auto_tagged']}</b>auto-tagged ({share:.0f}%)</div>
<div class="stat"><b>{s['needs_review']}</b>need review</div>
</div>
<div class="legend">{legend}</div>
<h2>To review ({len(review)})</h2>
<table><tr><th>#</th><th>Clause</th><th>Best guess</th><th>Conf.</th></tr>{review_list}</table>
<h2>All clauses</h2>
{body}
</main></body></html>
"""


def build_document(source: str, clauses: list[dict], tags: list[dict], labels: list[str], threshold: float) -> dict:
    tagged = [{**c, "label": t["label"], "confidence": t["confidence"], "needs_review": t["needs_review"],
               "runner_up": t.get("runner_up")} for c, t in zip(clauses, tags)]
    return {"source": source, "date": date.today().isoformat(), "threshold": threshold, "labels": labels,
            "summary": summarize(tagged, labels), "clauses": tagged}


def write_reports(doc: dict, out_stem: Path) -> list[Path]:
    """Write <stem>.json, <stem>.md and <stem>.html; return the paths."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, render in ((".json", to_json), (".md", to_markdown), (".html", to_html)):
        path = out_stem.with_name(out_stem.name + suffix)
        path.write_text(render(doc), encoding="utf-8")
        paths.append(path)
    return paths
