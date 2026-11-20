"""Milestone 5: merge the adapter into GPT-2 and tag new clauses.

Merge.  Each LoRALinear is replaced by its base layer with W' = W + (B @ A) * scale
(transposed for GPT-2's Conv1D). The merged model is an ordinary distilgpt2 with
k extra tokens: no LoRA code runs at inference, and its scores equal the unmerged
model's (checked in milestone 5 and in the tests).

Inference.  One forward pass per batch of clauses. At the end of each prompt
("clause text ") we score only the k label tokens:
    label = argmax over the k label-token scores   (the model can only answer a valid type)
    confidence = softmax over those k scores, at the chosen label
A clause whose confidence is below the review threshold keeps its best guess but
is marked NEEDS REVIEW, so a clerk looks at it instead of trusting it.
"""
from pathlib import Path

import numpy as np
import torch

from app.adapter import DEFAULT_ADAPTER, load_adapter
from app.lora import merge_lora
from app.metrics import softmax
from app.model import ClauseEncoder, label_logits

DEFAULT_THRESHOLD = 0.7          # replaced by the calibrated threshold saved in the adapter (standout 2)


def decide(probs: np.ndarray, labels: list[str], threshold: float) -> list[dict]:
    """Label, confidence, NEEDS REVIEW flag and runner-up for each row of label probabilities."""
    out = []
    for p in np.asarray(probs):
        order = np.argsort(-p)
        confidence = float(p[order[0]])
        out.append({
            "label": labels[order[0]],
            "confidence": round(confidence, 4),
            "needs_review": confidence < threshold,
            "runner_up": labels[order[1]] if len(order) > 1 else None,
            "probs": {labels[i]: round(float(p[i]), 4) for i in order},
        })
    return out


class ClauseTagger:
    def __init__(self, adapter: str | Path = DEFAULT_ADAPTER, merge: bool = True, threshold: float | None = None):
        self.model, self.tokenizer, self.label_ids, self.cfg = load_adapter(Path(adapter))
        if merge:
            merge_lora(self.model)
        self.labels = self.cfg.labels
        self.threshold = threshold if threshold is not None else self.cfg.extra.get("threshold", DEFAULT_THRESHOLD)
        self.encoder = ClauseEncoder(self.tokenizer, self.label_ids, self.cfg.max_len)

    @torch.no_grad()
    def scores(self, clauses: list[str]) -> np.ndarray:
        """(N, k) label-token logits."""
        prompts = self.encoder.prompts(clauses)
        return label_logits(self.model, prompts, self.label_ids, self.tokenizer.pad_token_id).numpy()

    def tag(self, clauses: list[str]) -> list[dict]:
        return decide(softmax(self.scores(clauses)), self.labels, self.threshold)
