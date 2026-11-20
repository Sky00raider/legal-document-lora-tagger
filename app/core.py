"""Your code. One stub per milestone; build them in order and keep each one small."""

# --- Milestone 1: Prepare a dataset of legal clauses for tagging
# Hint: Use the data preparation techniques from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_1(top_k: int = 6, per_class: int = 100, eval_per_class: int = 50, seed: int = 0) -> dict:
    """Download LEDGAR, keep the top_k clause types, write balanced train/val/test files."""
    from app.data import DATA_DIR, build_dataset, format_example, load_ledgar, save_dataset

    dataset = build_dataset(load_ledgar(), top_k, per_class, eval_per_class, seed)
    save_dataset(dataset)
    print(f"Label set ({top_k} clause types, count in LEDGAR train after dedupe):")
    for name, n in dataset["label_counts_in_ledgar_train"].items():
        print(f"  {name:<22} {n:>6}")
    print(f"train {len(dataset['train'])}, val {len(dataset['val'])}, test {len(dataset['test'])} -> {DATA_DIR}/")
    first = dataset["train"][0]
    print("example (end of training string):  ..." + format_example(first["text"], first["label"])[-90:])
    return dataset


# --- Milestone 2: Extend GPT-2 tokenizer and model for custom tags
# Hint: Apply the special token extension methods from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Implement LoRA adapter and inject into GPT-2
# Hint: Refer to the 'Build LoRA from Scratch' drill for implementing and injecting the LoRA adapter.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Fine-tune GPT-2 with LoRA on the clause dataset
# Hint: Combine the fine-tuning approach from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' with the LoRA injection from 'Build LoRA from Scratch'.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Merge LoRA adapter and perform inference on new clauses
# Hint: Follow the merging and inference steps detailed in the 'Build LoRA from Scratch' and 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drills.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Evaluate tagging accuracy and model efficiency
# Hint: Design evaluation metrics similar to those used in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill to assess performance.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
