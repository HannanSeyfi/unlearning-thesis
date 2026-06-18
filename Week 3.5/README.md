# Week 3.5: High-Accuracy Synthetic-Fact Baseline with Strict Scoring

This stage recreates the learned model that will be the starting point for
gradient-ascent unlearning. It uses the same strict scorer used by Week 4 and
Week 5.

## Baseline

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Synthetic dataset: `synthetic_facts_v1`
- Fictional identities: 100
- Unique facts: 500
- Forget facts: 100 from 20 identities
- Retain facts: 400 from 80 identities
- Training examples: 500
- Synthetic evaluation questions: 1,500
- Training-identical evaluation prompts: 500
- Held-out evaluation paraphrases: 1,000
- Fixed general-control questions: 50
- LoRA rank: 16
- LoRA alpha: 32
- Epochs: 20
- Learning rate: `3e-4`

The strict scorer uses deterministic generation with 16 new tokens and a
case-insensitive normalized whole-token boundary match. This avoids substring
false positives such as counting `56` inside `560`.

The archived reference run remains preserved under
`results/reference_successful_run`, but the current Week 4 and Week 5 notebooks
require the strict output folder below.

## Run

Open and run:

`notebooks/week3_5_train_high_accuracy_baseline_strict.ipynb`

New outputs are saved to:

`MyDrive/Thesis/Week 3.5/results/qwen05_high_accuracy_baseline`

If an adapter already exists and only the scoring files need to be refreshed,
run:

`notebooks/week3_5_strict_reevaluate_learned_model.ipynb`

Week 4 and Week 5 load the strict adapter from
`results/qwen05_high_accuracy_baseline/adapter` and require
`results/qwen05_high_accuracy_baseline/metrics.json` to contain the
`strict_scoring` metadata.
