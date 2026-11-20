# Sample input and tagged output

| File | What it is |
|---|---|
| `ivri_gcc_excerpt.txt` | 30 clauses from the General Conditions of Contract of ICAR-Indian Veterinary Research Institute ([source PDF](https://ivri.nic.in/tender/gcc.pdf), a public document). Text extracted from the PDF with `app/split.py`, numbering kept. |
| `ivri_gcc_excerpt.tagged.html` | The clerk report: open it in any browser (one self-contained file). |
| `ivri_gcc_excerpt.tagged.md` / `.json` | The same tagged document as Markdown (readable on GitHub) and JSON (for other software). |
| `imd_tender_td310.tagged.html` | A full 152-page real tender (India Meteorological Department), tagged end to end from the PDF. |

Re-create them (no training needed, under a minute on a laptop CPU):

```bash
python -m app.cli tag samples/ivri_gcc_excerpt.txt
python -m app.cli tag data/real_docs/imd_tender_td310.pdf --out samples/imd_tender_td310.tagged
```
