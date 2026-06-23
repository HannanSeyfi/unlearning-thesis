# Week 5 Aggressive Contrast Evaluation

This folder contains a Colab-ready evaluation run for an aggressive Week 5
checkpoint.

The default target is:

- candidate: `c09_most_aggressive`
- epoch: `3`
- adapter path:
  `Week 5/results/retain_regularized_unlearning_resumable_v1/resume_state/epoch_checkpoints/c09_most_aggressive/epoch_03/adapter`

This checkpoint was not selected by Week 5 because it failed the retain
threshold, but it is useful as a contrast point: stronger forgetting with much
larger retain damage.

## Output Safety

This run does not overwrite the main Week 5 result folder. It writes to:

`Week 5/results/aggressive_contrast_evaluation_v1/c09_most_aggressive_epoch_03`

The Colab notebook commits and pushes only that new run folder.

## Run In Colab

Open:

https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/main/Week%205/aggressive_contrast_evaluation/week5_aggressive_contrast_eval_colab.ipynb

Before running, add `GITHUB_TOKEN` in Colab Secrets and give the notebook access
to it. The final cell uses the existing repo helper to push the new output
folder back to GitHub.

## Run Locally

From the repository root:

```bash
python "Week 5/aggressive_contrast_evaluation/evaluate_week5_aggressive_contrast.py"
```

The local run expects the model dependencies used by the Week 5 notebooks:
`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, and `pandas`.

