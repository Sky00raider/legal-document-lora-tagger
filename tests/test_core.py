"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest
import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from app import data


# Milestone 1: Prepare a dataset of legal clauses for tagging
def _rows(spec: dict[str, int]) -> list[dict]:
    return [{"text": f"{label} clause number {i}.", "label": label}
            for label, n in spec.items() for i in range(n)]


def test_milestone_1_example_format_ends_with_one_label_token():
    assert data.label_token("Governing Laws") == "<governing_laws>"
    assert data.format_example("Governed by\n  New York   law.", "Governing Laws") == \
        "Governed by New York law. <governing_laws>"


def test_milestone_1_top_labels_and_balanced_sample():
    rows = _rows({"Notices": 30, "Waivers": 20, "Severability": 5})
    assert data.top_labels(rows, 2) == ["Notices", "Waivers"]
    sample = data.balanced_sample(rows, ["Notices", "Waivers"], per_class=10, seed=1)
    labels = [r["label"] for r in sample]
    assert labels.count("Notices") == 10 and labels.count("Waivers") == 10
    assert sample == data.balanced_sample(rows, ["Notices", "Waivers"], per_class=10, seed=1)


def test_milestone_1_dedupe_and_no_train_test_overlap():
    rows = [{"text": "Same  Clause.", "label": "A"}, {"text": "same clause.", "label": "B"},
            {"text": "Other.", "label": "A"}]
    assert [r["label"] for r in data.dedupe(rows)] == ["A", "A"]
    assert data.drop_overlap([{"text": "SAME clause.", "label": "A"}, {"text": "new", "label": "A"}],
                             rows) == [{"text": "new", "label": "A"}]


def test_milestone_1_build_dataset_on_fake_splits():
    splits = {"train": _rows({"A": 40, "B": 30, "C": 3}),
              "validation": _rows({"A": 2}) + [{"text": f"val {i}", "label": "B"} for i in range(9)],
              "test": [{"text": f"test {i}", "label": "A"} for i in range(9)]}
    ds = data.build_dataset(splits, top_k=2, per_class=10, eval_per_class=5)
    assert ds["labels"] == ["A", "B"] and len(ds["train"]) == 20
    assert all(r["text"].startswith("val") for r in ds["val"])   # copies of train text removed
    assert len(ds["test"]) == 5


# Shared tiny model: a 2-layer GPT-2 built from a config, random weights, no download.
def tiny_gpt2(vocab: int = 50, n_labels: int = 3):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=vocab, n_positions=32, n_embd=16, n_layer=2, n_head=2))
    model.resize_token_embeddings(vocab + n_labels)
    return model.eval(), list(range(vocab, vocab + n_labels))


# Milestone 2: Extend GPT-2 tokenizer and model for custom tags
def test_milestone_2_only_label_rows_get_gradient():
    from app.model import train_only_label_rows
    model, label_ids = tiny_gpt2()
    for p in model.parameters():
        p.requires_grad_(False)
    train_only_label_rows(model, label_ids)
    model(torch.tensor([[1, 2, 3, label_ids[0]]]), labels=torch.tensor([[1, 2, 3, label_ids[0]]])).loss.backward()
    grad = model.get_input_embeddings().weight.grad
    assert grad[label_ids].abs().sum() > 0
    assert grad[:label_ids[0]].abs().sum() == 0          # the original vocabulary never moves


def test_milestone_2_label_logits_match_full_forward():
    from app.model import label_logits
    model, label_ids = tiny_gpt2()
    prompts = [[1, 2, 3], [4, 5]]
    scores = label_logits(model, prompts, label_ids, pad_id=0)
    for i, p in enumerate(prompts):
        full = model(torch.tensor([p])).logits[0, -1]
        assert torch.allclose(scores[i], full[label_ids], atol=1e-5)


@pytest.mark.slow
def test_milestone_2_real_tokenizer_label_token_is_atomic():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from app.model import ClauseEncoder, add_label_tokens
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    ids = add_label_tokens(model, tok, ["Governing Laws", "Notices"])
    assert tok.tokenize("<governing_laws>") == ["<governing_laws>"] and tok.pad_token == tok.eos_token
    text = "This Agreement is governed by Delaware law."
    expected = tok(data.format_example(text, "Notices"), add_special_tokens=False).input_ids
    assert ClauseEncoder(tok, ids).training_examples([text], [1])[0] == expected


# Milestone 3: Implement LoRA adapter and inject into GPT-2
def test_milestone_3_lora_linear_shape_and_zero_init():
    from app.lora import LoRALinear
    base = nn.Linear(10, 6)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 5, 10)
    assert lora(x).shape == (3, 5, 6)
    assert torch.equal(lora(x), base(x))                 # B = 0, so nothing changes at init
    assert lora.A.shape == (4, 10) and lora.B.shape == (6, 4) and lora.scale == 2.0
    assert [n for n, p in lora.named_parameters() if p.requires_grad] == ["A", "B"]


def test_milestone_3_inject_freezes_everything_but_a_and_b():
    from app.lora import LoRALinear, count_parameters, inject_lora
    model, _ = tiny_gpt2()
    names = inject_lora(model, "c_attn", r=2, alpha=4)
    assert names == ["transformer.h.0.attn.c_attn", "transformer.h.1.attn.c_attn"]
    assert all(isinstance(model.get_submodule(n), LoRALinear) for n in names)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith((".A", ".B")) for n in trainable)
    lora, _ = count_parameters(model)
    assert lora == 2 * 2 * (16 + 48)                     # 2 layers x r x (in + out)


@pytest.mark.parametrize("make", [lambda: nn.Linear(8, 12), lambda: Conv1D(12, 8)])
def test_milestone_3_merge_equals_unmerged(make):
    from app.lora import LoRALinear
    torch.manual_seed(1)
    lora = LoRALinear(make(), r=4, alpha=8)
    nn.init.normal_(lora.B)                               # pretend it was trained
    x = torch.randn(2, 3, 8)
    before = lora(x)
    merged = lora.merged()
    assert torch.allclose(merged(x), before, atol=1e-5)


def test_milestone_3_conv1d_weight_is_transposed_on_merge():
    from app.lora import LoRALinear
    conv = Conv1D(12, 8)                                  # GPT-2 layer 8 -> 12, weight stored (8, 12)
    assert conv.weight.shape == (8, 12)
    lora = LoRALinear(conv, r=2, alpha=2)
    assert (lora.in_features, lora.out_features) == (8, 12)
    nn.init.normal_(lora.B)
    expected = conv.weight.detach().clone() + lora.delta_weight().T
    assert torch.allclose(lora.merged().weight, expected)


# Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
def test_milestone_4_loss_is_masked_to_the_label_position():
    from app.train import LabelPositionCollator
    batch = LabelPositionCollator(pad_id=0)([{"input_ids": [5, 6, 7, 99]}, {"input_ids": [8, 98]}])
    assert batch["input_ids"].tolist() == [[5, 6, 7, 99], [8, 98, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 99], [-100, 98, -100, -100]]


def test_milestone_4_training_step_moves_only_adapter_and_label_rows():
    from app.lora import inject_lora
    from app.model import train_only_label_rows
    from app.train import LabelPositionCollator
    model, label_ids = tiny_gpt2()
    inject_lora(model, r=2, alpha=4)
    train_only_label_rows(model, label_ids)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        batch = LabelPositionCollator(0)([{"input_ids": [1, 2, 3, label_ids[1]]}, {"input_ids": [4, label_ids[0]]}])
        model(**batch).loss.backward()
        opt.step()
        opt.zero_grad()
    for n, p in model.named_parameters():
        changed = not torch.equal(p, before[n])
        if n.endswith(".B"):
            assert changed, n
        elif n == "transformer.wte.weight":
            assert torch.equal(p[:label_ids[0]], before[n][:label_ids[0]])
            assert not torch.equal(p[label_ids], before[n][label_ids])
        elif not n.endswith(".A"):
            assert not changed, n


def test_milestone_4_adapter_round_trip(tmp_path):
    from app.adapter import TaggerConfig, load_into, save_adapter
    from app.lora import inject_lora
    from app.model import label_logits
    model, label_ids = tiny_gpt2()
    inject_lora(model, r=2, alpha=4)
    for n, p in model.named_parameters():
        if n.endswith(".B"):
            nn.init.normal_(p)
    with torch.no_grad():
        model.get_input_embeddings().weight[label_ids] += 1.0
    size = save_adapter(model, label_ids, TaggerConfig(labels=["a", "b", "c"], r=2, alpha=4), tmp_path / "a.pt")
    assert size > 0
    fresh, _ = tiny_gpt2()                               # same seed: same "pretrained" base
    inject_lora(fresh, r=2, alpha=4)
    load_into(fresh, label_ids, torch.load(tmp_path / "a.pt", weights_only=True))
    prompts = [[1, 2, 3], [4, 5, 6, 7]]
    assert torch.allclose(label_logits(model, prompts, label_ids, 0), label_logits(fresh, prompts, label_ids, 0))


# Milestone 5: Merge LoRA adapter and perform inference on new clauses
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Evaluate tagging accuracy and model efficiency
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")

