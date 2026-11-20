## LEDGAR test set

**Accuracy 97.0%, macro-F1 97.0%** on 300 held-out clauses (6 clause types, 50 each).

| Clause type | Precision | Recall | F1 | Test clauses |
|---|---:|---:|---:|---:|
| Governing Laws | 1.00 | 0.98 | 0.99 | 50 |
| Notices | 0.98 | 0.98 | 0.98 | 50 |
| Counterparts | 0.96 | 1.00 | 0.98 | 50 |
| Entire Agreements | 0.98 | 0.94 | 0.96 | 50 |
| Severability | 0.96 | 0.94 | 0.95 | 50 |
| Amendments | 0.94 | 0.98 | 0.96 | 50 |

Most confused pairs (true -> predicted, both directions):

- Entire Agreements <-> Amendments: 1 + 1
- Governing Laws <-> Severability: 1 + 0
- Entire Agreements <-> Severability: 1 + 0
- Notices <-> Severability: 0 + 1
- Severability <-> Amendments: 1 + 0

## Efficiency

| | |
|---|---:|
| Parameters in distilgpt2 | 82,064,640 |
| Trainable (LoRA A/B + 6 label rows) | 152,064 (0.19%) |
| LoRA A/B only | 147,456 (0.18%) |
| Adapter file (what we ship) | 599 KB |
| Full fine-tuned checkpoint (float32) | 313 MB |
| Adapter / full model | 1 : 535 |
| Training time (laptop CPU, 600 clauses x 6 epochs) | 37.8 min |
