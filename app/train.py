"""Milestone 4: fine-tune distilgpt2 with LoRA on the clause dataset, on a laptop CPU.

Model:  distilgpt2 (6 layers, 82M parameters) + one token per clause type
        (milestone 2) + LoRA r=8, alpha=16 on every attention c_attn (milestone 3).
        Trainable: the LoRA A/B matrices and the k new label rows; everything else frozen.

Loss masked to the label position.  Each example is "clause text <label>" as token ids.
`labels` is -100 (ignored) everywhere except the final position, which holds the label
token. GPT-2 shifts labels by one internally, so the only prediction scored is the one
made at the last prompt token: "given this clause, which label comes next?". The model
is not trained to reproduce contract text, only to name its type.

Training uses the Hugging Face Trainer: lr 1e-3, 6 epochs, batch 8, no weight decay
(so the frozen embedding rows can never drift, see app/model.py). On a laptop short of RAM,
grad_accum=2 runs two half-batches of 4 and adds their gradients: the same update, half the peak memory.
"""
import time
from dataclasses import dataclass

import torch

from app.adapter import TaggerConfig, build
from app.lora import count_parameters
from app.model import ClauseEncoder, label_logits, train_only_label_rows


@dataclass
class LabelPositionCollator:
    """Right-pads a batch; labels are -100 everywhere except the final (label token) position."""
    pad_id: int

    def __call__(self, features: list[dict]) -> dict:
        seqs = [f["input_ids"] for f in features]
        T = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), T), dtype=torch.long)
        labels = torch.full((len(seqs), T), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s)
            attention_mask[i, :len(s)] = 1
            labels[i, len(s) - 1] = s[-1]          # the label token, nothing else is scored
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def accuracy(model, encoder: ClauseEncoder, rows: list[dict], labels: list[str], pad_id: int) -> float:
    scores = label_logits(model, encoder.prompts([r["text"] for r in rows]), encoder.label_ids, pad_id)
    target = torch.tensor([labels.index(r["label"]) for r in rows])
    return (scores.argmax(dim=1) == target).float().mean().item()


def train(cfg: TaggerConfig, train_rows: list[dict], val_rows: list[dict], epochs: float = 6,
          lr: float = 1e-3, batch_size: int = 8, seed: int = 0, out_dir: str = "data/trainer",
          full_finetune: bool = False, grad_accum: int = 1) -> dict:
    """Train and return the model, tokenizer, label ids and a small report.

    full_finetune=True trains every weight without LoRA (for the "why LoRA" comparison).
    """
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(seed)
    model, tokenizer, label_ids = build(cfg, lora=not full_finetune)
    if full_finetune:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        train_only_label_rows(model, label_ids)
    encoder = ClauseEncoder(tokenizer, label_ids, cfg.max_len)

    lora_params, total = count_parameters(model)
    trainable = (sum(p.numel() for p in model.parameters() if p.requires_grad) if full_finetune
                 else lora_params + len(label_ids) * model.config.n_embd)
    print(f"parameters: {total:,} total | LoRA A/B {lora_params:,} ({100 * lora_params / total:.2f}%) | "
          f"trainable incl. {len(label_ids)} label rows {trainable:,} ({100 * trainable / total:.2f}%)")

    targets = [cfg.labels.index(r["label"]) for r in train_rows]
    examples = [{"input_ids": ids} for ids in encoder.training_examples([r["text"] for r in train_rows], targets)]
    args = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size // grad_accum, gradient_accumulation_steps=grad_accum,
        weight_decay=0.0, lr_scheduler_type="linear",
        logging_steps=25, save_strategy="no", report_to="none", use_cpu=True, seed=seed,
        remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=LabelPositionCollator(tokenizer.pad_token_id))
    start = time.time()
    result = trainer.train()
    seconds = time.time() - start

    val_acc = accuracy(model, encoder, val_rows, cfg.labels, tokenizer.pad_token_id)
    report = {"train_seconds": round(seconds, 1), "final_train_loss": round(result.training_loss, 4),
              "val_accuracy": round(val_acc, 4), "trainable_params": trainable, "lora_params": lora_params,
              "total_params": total, "epochs": epochs, "lr": lr, "batch_size": batch_size,
              "train_examples": len(train_rows), "loss_log": [
                  {"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h]}
    print(f"trained in {seconds / 60:.1f} min, final train loss {result.training_loss:.4f}, "
          f"val accuracy {val_acc:.3f}")
    return {"model": model, "tokenizer": tokenizer, "label_ids": label_ids, "encoder": encoder, "report": report}
