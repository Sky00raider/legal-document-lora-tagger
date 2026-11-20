r"""Milestone 3: LoRA written by hand (no peft library).

A pretrained layer computes  y = x W + b  with a big frozen weight W. Full
fine-tuning would change all of W. LoRA instead learns a small correction with the
same shape as W, written as the product of two thin matrices:

    y = base(x)  +  scale * (x @ A.T) @ B.T            scale = alpha / r
                    \___ A is (r x in), B is (out x r): the only trainable part ___/

GPT-2's attention projection `c_attn` maps 768 -> 2304 (query, key and value at
once): W has 1.77M numbers, while A and B with r = 8 have 8 * (768 + 2304) = 24,576.

Initialisation (as in the LoRA paper and the course drill):
  * A small random (randn * 0.01), B all zeros  ->  B @ A = 0, so at the start the
    wrapped model computes exactly what the pretrained model computed.
  * B still gets a gradient on the first step (it depends on x A^T, which is not
    zero), so learning starts immediately; after that A learns too.
  * scale = alpha / r keeps the size of the update similar when r changes.

GPT-2's Conv1D.  Hugging Face's GPT-2 does not use nn.Linear: its `Conv1D` stores the
weight transposed, as (in, out), and computes x @ W + b. `forward` never touches W
(it calls the base layer and adds the LoRA term), so it works for both. Only merging
needs to know: the update B @ A is (out, in), so for Conv1D we add its transpose.

Merging.  After training, W' = W + scale * B @ A  (transposed for Conv1D) gives one
plain layer with the adapter baked in: same outputs, no extra cost at inference.
"""
import torch
from torch import nn


def is_conv1d(layer: nn.Module) -> bool:
    return type(layer).__name__ == "Conv1D"     # transformers.pytorch_utils.Conv1D


def layer_shape(layer: nn.Module) -> tuple[int, int]:
    """(in_features, out_features) of an nn.Linear or a GPT-2 Conv1D."""
    if isinstance(layer, nn.Linear):
        return layer.in_features, layer.out_features
    if is_conv1d(layer):
        n_in, n_out = layer.weight.shape        # stored (in, out)
        return n_in, n_out
    raise TypeError(f"LoRA does not know how to wrap {type(layer).__name__}")


class LoRALinear(nn.Module):
    """A frozen base layer plus a trainable low-rank update B @ A."""

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.in_features, self.out_features = layer_shape(base)
        self.r = r
        self.scale = alpha / r
        w = base.weight
        self.A = nn.Parameter(torch.randn(r, self.in_features, dtype=w.dtype, device=w.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.out_features, r, dtype=w.dtype, device=w.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def delta_weight(self) -> torch.Tensor:
        """The (out x in) matrix this adapter adds: scale * B @ A."""
        return self.scale * (self.B @ self.A)

    @torch.no_grad()
    def merged(self) -> nn.Module:
        """The base layer with the update baked into its weight (W' = W + delta; delta.T for Conv1D)."""
        delta = self.delta_weight()
        self.base.weight += delta.T if is_conv1d(self.base) else delta
        return self.base

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, scale={self.scale:g}"


def _replace(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, child, new)


def inject_lora(model: nn.Module, target: str = "c_attn", r: int = 8, alpha: float = 16.0) -> list[str]:
    """Freeze every parameter, then wrap each layer named `target` in a LoRALinear.

    Returns the full names of the wrapped layers.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in model.named_modules() if n == target or n.endswith("." + target)]
    if not names:
        raise ValueError(f"no layer called {target!r} in the model")
    for name in names:          # collected first: never modify modules while iterating them
        _replace(model, name, LoRALinear(model.get_submodule(name), r, alpha))
    return names


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear by its merged base layer. Returns how many were merged."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        _replace(model, name, model.get_submodule(name).merged())
    return len(names)


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every LoRA A and B in the model, by parameter name, copied to CPU."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(LoRA parameters, all parameters). Shared/tied tensors are counted once."""
    seen, lora, total = set(), 0, 0
    for n, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if n.endswith((".A", ".B")):
            lora += p.numel()
    return lora, total
