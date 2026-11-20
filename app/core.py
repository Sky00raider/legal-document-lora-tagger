"""Your code. One stub per milestone; build them in order and keep each one small."""

# --- Milestone 1: Prepare a dataset of legal clauses for tagging
# Hint: Use the data preparation techniques from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_1(run: str = "top6", per_class: int = 100, eval_per_class: int = 50, seed: int = 0) -> dict:
    """Download LEDGAR, keep the run's clause types, write balanced train/val/test files."""
    from app.data import build_dataset, format_example, load_ledgar, save_dataset
    from app.runs import get_run

    r = get_run(run)
    dataset = build_dataset(load_ledgar(), r.top_k, per_class, eval_per_class, seed, r.labels)
    save_dataset(dataset, r.data_dir)
    print(f"Label set '{run}' ({len(dataset['labels'])} clause types, count in LEDGAR train after dedupe):")
    for name, n in dataset["label_counts_in_ledgar_train"].items():
        print(f"  {name:<22} {n:>6}")
    print(f"train {len(dataset['train'])}, val {len(dataset['val'])}, test {len(dataset['test'])} -> {r.data_dir}/")
    first = dataset["train"][0]
    print("example (end of training string):  ..." + format_example(first["text"], first["label"])[-90:])
    return dataset


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags
# Hint: Apply the special token extension methods from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_2(run: str = "top6", base: str = "distilbert/distilgpt2") -> None:
    """Add one atomic token per clause type and show what changed."""
    from transformers import AutoTokenizer

    from app.data import label_token, load_labels, load_split
    from app.model import ClauseEncoder, load_base
    from app.runs import get_run

    data_dir = get_run(run).data_dir
    labels = load_labels(data_dir)
    before = AutoTokenizer.from_pretrained(base)
    model, tokenizer, label_ids = load_base(base, labels)
    token = label_token(labels[0])
    print(f"vocab {len(before):,} -> {len(tokenizer):,} (+{len(labels)} label tokens), pad token = {tokenizer.pad_token!r}")
    print(f"embedding matrix: {tuple(model.get_input_embeddings().weight.shape)}, "
          f"output layer tied to it: {model.get_output_embeddings().weight is model.get_input_embeddings().weight}")
    print(f"{token!r} before: {before.tokenize(token)}  after: {tokenizer.tokenize(token)}")
    for name, i in zip(labels, label_ids):
        print(f"  {label_token(name):<24} id {i}")
    row = load_split("train", data_dir)[0]
    ids = ClauseEncoder(tokenizer, label_ids).training_examples([row["text"]], [labels.index(row["label"])])[0]
    print(f"a training example is {len(ids)} tokens; its last 4 tokens: {[tokenizer.decode([i]) for i in ids[-4:]]}")


# --- Milestone 3: Implement LoRA adapter and inject into GPT-2
# Hint: Refer to the 'Build LoRA from Scratch' drill for implementing and injecting the LoRA adapter.
def milestone_3(run: str = "top6", base: str = "distilbert/distilgpt2", r: int = 8, alpha: float = 16.0) -> None:
    """Freeze everything, wrap every c_attn in a LoRALinear, check nothing changed yet."""
    import torch

    from app.data import load_labels, load_split
    from app.lora import count_parameters, inject_lora
    from app.model import ClauseEncoder, label_logits, load_base
    from app.runs import get_run

    data_dir = get_run(run).data_dir
    model, tokenizer, label_ids = load_base(base, load_labels(data_dir))
    prompts = ClauseEncoder(tokenizer, label_ids).prompts([row["text"] for row in load_split("val", data_dir)[:8]])
    before = label_logits(model, prompts, label_ids, tokenizer.pad_token_id)

    wrapped = inject_lora(model, "c_attn", r, alpha)
    lora, total = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"wrapped {len(wrapped)} layers: {wrapped[0]} ... {wrapped[-1]}")
    print(model.get_submodule(wrapped[0]))
    print(f"total {total:,} | trainable {trainable:,} ({100 * trainable / total:.3f}%)  [r={r}, alpha={alpha}]")
    print(f"per layer: A {tuple(model.get_submodule(wrapped[0]).A.shape)} + "
          f"B {tuple(model.get_submodule(wrapped[0]).B.shape)} = {lora // len(wrapped):,} numbers")
    after = label_logits(model, prompts, label_ids, tokenizer.pad_token_id)
    print(f"B = 0 at init, so outputs are unchanged: max |difference| = {(after - before).abs().max():.2e}"
          f" -> {torch.allclose(before, after)}")


# --- Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
# Hint: Combine the fine-tuning approach from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' with the LoRA injection from 'Build LoRA from Scratch'.
def milestone_4(run: str = "top6", base: str = "distilbert/distilgpt2", r: int = 8, alpha: float = 16.0,
                epochs: float = 6, grad_accum: int = 1) -> dict:
    """Train, save the adapter, reload it onto a fresh base and check accuracy matches."""
    import json
    from pathlib import Path

    from app.adapter import TaggerConfig, load_adapter, save_adapter
    from app.data import load_labels, load_split
    from app.model import ClauseEncoder
    from app.runs import get_run
    from app.train import accuracy, train

    spec = get_run(run)
    adapter_path = spec.adapter
    cfg = TaggerConfig(labels=load_labels(spec.data_dir), base=base, r=r, alpha=alpha)
    cfg.extra["run"] = run
    val = load_split("val", spec.data_dir)
    out = train(cfg, load_split("train", spec.data_dir), val, epochs=epochs, grad_accum=grad_accum)
    report = out["report"]
    cfg.extra["training"] = {k: v for k, v in report.items() if k != "loss_log"}
    size = save_adapter(out["model"], out["label_ids"], cfg, Path(adapter_path))
    print(f"adapter saved: {adapter_path} ({size / 1024:.0f} KB)")

    model, tokenizer, label_ids, _ = load_adapter(Path(adapter_path))
    reloaded = accuracy(model, ClauseEncoder(tokenizer, label_ids, cfg.max_len), val, cfg.labels,
                        tokenizer.pad_token_id)
    print(f"reloaded onto a fresh {base}: val accuracy {reloaded:.4f} (trained model: {report['val_accuracy']:.4f})"
          f" -> {'match' if abs(reloaded - report['val_accuracy']) < 1e-6 else 'MISMATCH'}")
    report.update(adapter_bytes=size, reloaded_val_accuracy=round(reloaded, 4))
    spec.out.mkdir(parents=True, exist_ok=True)
    (spec.out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --- Milestone 5: Merge LoRA adapter and perform inference on new clauses
# Hint: Follow the merging and inference steps detailed in the 'Build LoRA from Scratch' and 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drills.
NEW_CLAUSES = [   # written for this demo, not from LEDGAR
    "This Agreement shall be governed by and construed in accordance with the laws of India, and the courts at "
    "New Delhi shall have exclusive jurisdiction.",
    "If any provision of this Agreement is held invalid or unenforceable, the remaining provisions shall remain "
    "in full force and effect.",
    "All notices under this Agreement shall be in writing and delivered by registered post or e-mail to the "
    "addresses set out above.",
    "The Supplier shall not assign or transfer this contract or any part of it without the prior written consent "
    "of the Purchaser.",
    "The Supplier shall deliver 34,400 radiosondes to the consignee within 120 days of the date of the contract.",
]


def milestone_5(run: str = "top6") -> None:
    """Merge W' = W + (B@A)*scale into GPT-2, check the scores are unchanged, tag new clauses."""
    from pathlib import Path

    from app.adapter import load_adapter
    from app.data import load_split
    from app.lora import LoRALinear, merge_lora
    from app.metrics import softmax
    from app.model import ClauseEncoder, label_logits
    from app.runs import get_run
    from app.tagger import decide

    spec = get_run(run)
    model, tokenizer, label_ids, cfg = load_adapter(Path(spec.adapter))
    encoder = ClauseEncoder(tokenizer, label_ids, cfg.max_len)
    prompts = encoder.prompts([r["text"] for r in load_split("val", spec.data_dir)])
    before = label_logits(model, prompts, label_ids, tokenizer.pad_token_id)
    merged = merge_lora(model)
    after = label_logits(model, prompts, label_ids, tokenizer.pad_token_id)
    left = sum(isinstance(m, LoRALinear) for m in model.modules())
    print(f"merged {merged} LoRA layers into c_attn (LoRA layers left: {left})")
    print(f"label scores on {len(prompts)} val clauses, unmerged vs merged: max |difference| = "
          f"{(after - before).abs().max():.2e}, same predictions: "
          f"{bool((after.argmax(1) == before.argmax(1)).all())}")

    threshold = cfg.extra.get("threshold", 0.7)
    tags = decide(softmax(label_logits(model, encoder.prompts(NEW_CLAUSES), label_ids,
                                       tokenizer.pad_token_id).numpy()), cfg.labels, threshold)
    print(f"\nnew clauses (review threshold {threshold:.2f}):")
    for text, t in zip(NEW_CLAUSES, tags):
        flag = "NEEDS REVIEW" if t["needs_review"] else "ok"
        print(f"  {text[:85]}...\n    -> {t['label']} ({t['confidence']:.2f}, {flag}; runner-up {t['runner_up']})")


# --- Milestone 6: Evaluate tagging accuracy and model efficiency
# Hint: Design evaluation metrics similar to those used in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill to assess performance.
def milestone_6(run: str = "top6") -> dict:
    """Test-set accuracy, macro-F1, per-class results, confusions and efficiency numbers."""
    from app.evaluate import evaluate_ledgar
    return evaluate_ledgar(run)
