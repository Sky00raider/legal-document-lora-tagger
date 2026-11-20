"""Milestone 2: teach GPT-2's tokenizer and embeddings one new token per clause type.

  * Pad token.  GPT-2 has none, so the end-of-text token doubles as padding. Padding
    sits on the right, is masked out of attention, and is never a training target.

  * Label tokens.  "<governing_laws>", "<notices>", ... are added as special tokens, so
    the tokenizer never splits them: each clause type is exactly one token. The
    embedding matrix grows from V to V + k rows (`resize_token_embeddings`). GPT-2
    ties its output layer to the input embeddings, so each new row is also the vector
    the final hidden state is scored against when the model predicts the label.

  * Starting rows.  Each new row starts as the mean embedding of the label's English
    name (" Governing Laws" -> its ordinary GPT-2 tokens), not random noise: the
    pretrained model already knows what those words mean.

  * Training only the new rows.  The embedding matrix is one tensor, so we cannot
    freeze half of it. Instead `train_only_label_rows` sets requires_grad on it and adds
    a gradient hook that zeroes the gradient of the V original rows. With no weight
    decay on it (the Trainer default), the original rows never change; only the k
    label rows learn.

Encoding.  A training example is the string "clause text <label>" (app/data.py). In
token ids that is   clause tokens + [" "] + [label token]:   the space before a special
token becomes its own GPT-2 token. So the prompt is "clause text " and the model's job
is to predict the label token right after it. Long clauses keep their beginning (where
headings like "Governing Law." sit); the trailing space and label are always kept.
"""
import torch

from app.data import label_token

PROMPT_SUFFIX = " "


def add_label_tokens(model, tokenizer, labels: list[str]) -> list[int]:
    """Pad-token fix, one new token per label, resized embeddings. Returns the label token ids."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    embeddings = model.get_input_embeddings().weight
    with torch.no_grad():
        init = torch.stack([embeddings[tokenizer(" " + name, add_special_tokens=False).input_ids].mean(dim=0)
                            for name in labels])

    tokens = [label_token(name) for name in labels]
    added = tokenizer.add_tokens(tokens, special_tokens=True)
    if added != len(labels):
        raise ValueError(f"expected {len(labels)} new label tokens, the tokenizer added {added}")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    label_ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        model.get_input_embeddings().weight[label_ids] = init.to(embeddings.dtype)
    return label_ids


def load_base(base: str, labels: list[str]):
    """Pretrained GPT-2 + tokenizer with the label tokens added. Returns (model, tokenizer, label_ids)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    return model, tokenizer, add_label_tokens(model, tokenizer, labels)


def train_only_label_rows(model, label_ids: list[int]):
    """Make the embedding matrix trainable, but zero the gradient of every row except the label rows."""
    weight = model.get_input_embeddings().weight
    weight.requires_grad_(True)
    keep = torch.zeros(weight.shape[0], 1, dtype=weight.dtype, device=weight.device)
    keep[label_ids] = 1.0
    return weight.register_hook(lambda grad: grad * keep)


class ClauseEncoder:
    """Clause text -> token ids, for training (with label) and prediction (without)."""

    def __init__(self, tokenizer, label_ids: list[int], max_len: int = 256):
        self.tokenizer = tokenizer
        self.label_ids = list(label_ids)
        self.suffix_ids = tokenizer(PROMPT_SUFFIX, add_special_tokens=False).input_ids
        self.room = max_len - len(self.suffix_ids) - 1      # clause tokens that fit
        if self.room < 1:
            raise ValueError(f"max_len={max_len} leaves no room for the clause")

    def prompts(self, texts: list[str]) -> list[list[int]]:
        """'clause text ' as ids, clause cut to fit."""
        clean = [" ".join(t.split()) for t in texts]
        ids = self.tokenizer(clean, add_special_tokens=False, verbose=False).input_ids
        return [x[:self.room] + self.suffix_ids for x in ids]

    def training_examples(self, texts: list[str], targets: list[int]) -> list[list[int]]:
        """'clause text <label>' as ids: the prompt followed by the label token of each class index."""
        return [p + [self.label_ids[t]] for p, t in zip(self.prompts(texts), targets)]


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad id lists into input_ids (B, T) and attention_mask (B, T)."""
    T = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        attention_mask[i, :len(s)] = 1
    return input_ids, attention_mask


@torch.no_grad()
def label_logits(model, prompts: list[list[int]], label_ids: list[int], pad_id: int,
                 batch_size: int = 32) -> torch.Tensor:
    """One forward pass per batch: the scores of the k label tokens after each prompt, (N, k).

    We take the hidden state at each prompt's last real token and score it against the
    k label rows of the (tied) output layer only, so we never build the full
    (N, T, V) vocabulary logits. Prompts are batched by length to waste less padding.
    """
    model.eval()
    rows = model.get_output_embeddings().weight[label_ids]          # (k, d)
    out = torch.zeros(len(prompts), len(label_ids))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        input_ids, attention_mask = pad_batch([prompts[i] for i in idx], pad_id)
        hidden = model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1
        out[idx] = hidden[torch.arange(len(idx)), last] @ rows.T
    return out
