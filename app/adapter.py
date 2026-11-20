"""Build the LoRA clause tagger, and save / reload only the parts we trained.

The pretrained base (distilgpt2, ~330 MB) is never saved: anyone can download it
again. What makes the tagger ours is small, and goes into one file:
  * every LoRA A and B matrix,
  * the k new label-token embedding rows,
  * the settings needed to rebuild the frozen part (base model, labels, r, alpha...).

Loading = rebuild a fresh tagger from the settings (pretrained base + label tokens +
LoRA layers with fresh random A and zero B), then overwrite the trained tensors with
the saved ones. If that is right, the reloaded model gives exactly the same scores.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from app.lora import inject_lora, lora_state
from app.model import load_base

FORMAT_VERSION = 1
DEFAULT_ADAPTER = Path("models/adapter.pt")


@dataclass
class TaggerConfig:
    labels: list[str]
    base: str = "distilbert/distilgpt2"
    r: int = 8
    alpha: float = 16.0
    target: str = "c_attn"
    max_len: int = 256
    extra: dict = field(default_factory=dict)       # threshold, training notes, ...


def build(cfg: TaggerConfig, lora: bool = True):
    """Pretrained base + label tokens (+ LoRA). Returns (model, tokenizer, label_ids)."""
    model, tokenizer, label_ids = load_base(cfg.base, cfg.labels)
    if lora:
        inject_lora(model, cfg.target, cfg.r, cfg.alpha)
    return model, tokenizer, label_ids


def save_adapter(model, label_ids: list[int], cfg: TaggerConfig, path: Path = DEFAULT_ADAPTER) -> int:
    """Write settings + LoRA tensors + label rows to one file; return its size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "config": asdict(cfg),
        "lora": lora_state(model),
        "label_ids": list(label_ids),
        "label_rows": model.get_input_embeddings().weight[label_ids].detach().cpu().clone(),
    }, path)
    return path.stat().st_size


@torch.no_grad()
def load_into(model, label_ids: list[int], saved: dict) -> None:
    """Copy saved tensors into an already-built model, checking every name and shape."""
    if list(label_ids) != saved["label_ids"]:
        raise ValueError("label token ids differ from the saved adapter: different base model or labels?")
    params = dict(model.named_parameters())
    for name, tensor in saved["lora"].items():
        if name not in params or params[name].shape != tensor.shape:
            raise ValueError(f"adapter tensor {name} {tuple(tensor.shape)} does not fit this model")
        params[name].copy_(tensor)
    model.get_input_embeddings().weight[label_ids] = saved["label_rows"]


def load_adapter(path: Path = DEFAULT_ADAPTER):
    """Fresh pretrained base + saved adapter. Returns (model, tokenizer, label_ids, cfg)."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unknown adapter format {saved['format_version']}")
    cfg = TaggerConfig(**saved["config"])
    model, tokenizer, label_ids = build(cfg)
    load_into(model, label_ids, saved)
    return model.eval(), tokenizer, label_ids, cfg
