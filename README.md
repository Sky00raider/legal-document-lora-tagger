# Legal Document LoRA Tagger

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/nwzyej8pnw/badge.svg)](https://ask.thili.ai/b/nwzyej8pnw)

This project develops a Legal Document LoRA Tagger, a tool designed to help public sector clerks classify documents faster. By fine-tuning GPT-2 with a LoRA adapter, it efficiently tags legal clauses in public records, aiming to improve efficiency and reduce manual review time for public services.

## What you are building

**Goal.** You will build a tool that uses LoRA to fine-tune GPT-2, enabling it to tag legal clauses in public records, which will result in a tagged legal document.

**Done when.** The project is complete when you have a fine-tuned GPT-2 model with a LoRA adapter that can classify legal contract clauses by type, demonstrating the technique from lfllm-lflegalgpt for fine-tuning and lfllm-lflora for efficient fine-tuning, with a final output being a tagged legal document.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a dataset of legal clauses for tagging**  
      Hint: Use the data preparation techniques from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **2. Extend GPT-2 tokenizer and model for custom tags**  
      Hint: Apply the special token extension methods from the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill.
- [ ] **3. Implement LoRA adapter and inject into GPT-2**  
      Hint: Refer to the 'Build LoRA from Scratch' drill for implementing and injecting the LoRA adapter.
- [ ] **4. Fine-tune GPT-2 with LoRA on the clause dataset**  
      Hint: Combine the fine-tuning approach from 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' with the LoRA injection from 'Build LoRA from Scratch'.
- [ ] **5. Merge LoRA adapter and perform inference on new clauses**  
      Hint: Follow the merging and inference steps detailed in the 'Build LoRA from Scratch' and 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drills.
- [ ] **6. Evaluate tagging accuracy and model efficiency**  
      Hint: Design evaluation metrics similar to those used in the 'Fine-Tune GPT-2 to Tag Legal Contract Clauses' drill to assess performance.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Compare the model's tagging efficiency and accuracy against a simulated manual review process.
- Explore different ways to visualize or present the tagged output to a clerk for review.
- Investigate if the model can be adapted to tag different types of legal documents or clauses with minimal changes.
