## LEDGAR test set

**Accuracy 95.8%, macro-F1 95.8%** on 697 held-out clauses (14 clause types, 50 each).

| Clause type | Precision | Recall | F1 | Test clauses |
|---|---:|---:|---:|---:|
| Governing Laws | 1.00 | 0.98 | 0.99 | 50 |
| Notices | 0.96 | 0.96 | 0.96 | 50 |
| Amendments | 0.94 | 1.00 | 0.97 | 50 |
| Severability | 0.96 | 0.94 | 0.95 | 50 |
| Entire Agreements | 0.96 | 0.98 | 0.97 | 50 |
| Terminations | 0.94 | 0.96 | 0.95 | 50 |
| Assignments | 0.96 | 0.96 | 0.96 | 50 |
| Confidentiality | 0.98 | 0.98 | 0.98 | 50 |
| Indemnifications | 0.98 | 0.90 | 0.94 | 50 |
| Taxes | 0.94 | 0.98 | 0.96 | 50 |
| Payments | 0.96 | 0.88 | 0.92 | 50 |
| Insurances | 0.92 | 0.98 | 0.95 | 50 |
| Arbitration | 0.98 | 1.00 | 0.99 | 47 |
| Warranties | 0.94 | 0.92 | 0.93 | 50 |

Most confused pairs (true -> predicted, both directions):

- Taxes <-> Payments: 1 + 3
- Assignments <-> Indemnifications: 1 + 1
- Amendments <-> Severability: 0 + 2
- Notices <-> Terminations: 1 + 1
- Confidentiality <-> Warranties: 1 + 1

## Efficiency

| | |
|---|---:|
| Parameters in distilgpt2 | 82,070,784 |
| Trainable (LoRA A/B + 14 label rows) | 158,208 (0.19%) |
| LoRA A/B only | 147,456 (0.18%) |
| Adapter file (what we ship) | 623 KB |
| Full fine-tuned checkpoint (float32) | 313 MB |
| Adapter / full model | 1 : 514 |
| Training time (laptop CPU, 1400 clauses x 6 epochs) | 83.6 min |
