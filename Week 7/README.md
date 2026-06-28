# Week 7: Adaptive Constrained Unlearning

Week 7 tests whether the Week 6 trade-off can be improved by adapting the
optimization pressure during training instead of using one fixed gradient
projection rule.

Week 6 preserved useful behavior, but its selected checkpoint reached only
`64.0%` forget held-out accuracy while retaining `87.0%`. The PCGrad-style
projection helped modestly on the selection split, yet it remained well above
the intended forgetting target.

Week 7 therefore asks:

Can forgetting pressure rise automatically while retain and general behavior
remain above explicit performance floors?

## Method

Every candidate starts from the same strict Week 3.5 Qwen 0.5B LoRA adapter.
After each epoch, the runner measures held-out forget, retain, and general
selection performance. An adaptive controller then:

- increases forget pressure when the constraints are satisfied and the forget
  target is still unmet;
- reduces forget pressure after a retain or general constraint violation;
- updates a non-negative dual variable that strengthens retain cross-entropy
  and KL preservation when the measured constraint gap grows.

The focused run contains three adaptive candidates with retain floors of
`82%`, `83%`, and `85%`, plus a matched fixed-pressure control. The best
adaptive and fixed candidates both receive the complete final evaluation.

## Targets

- forget held-out accuracy at or below `45%`;
- retain held-out accuracy at or above `82%` globally;
- general-control accuracy at or above `50%`.

Lower forget accuracy is better. Higher retain and general accuracy are
better.

## Run In Colab

Open:

https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/main/Week%207/notebooks/week7_adaptive_constrained_unlearning_colab.ipynb

Before running, add `GITHUB_TOKEN` in Colab Secrets and grant the notebook
access to it. Use a GPU runtime. The notebook clones the repository, resumes an
existing Week 7 run when present, and pushes rolling checkpoints and final
outputs back to GitHub. Its sparse checkout avoids downloading the large Week 6
checkpoint tree while retaining every input needed by Week 7.

## Run Locally

From the repository root:

```bash
python "Week 7/adaptive_constrained_unlearning/train_week7_adaptive_constrained_unlearning.py"
```

The dependencies are the same as Week 6: `torch`, `transformers`, `peft`,
`bitsandbytes`, `accelerate`, `pandas`, `numpy`, and `safetensors`.

## Outputs

The default run writes to:

`Week 7/results/adaptive_constrained_unlearning_v1`

Important outputs:

- `best_week7_adapter/`: globally selected Week 7 adapter
- `candidate_adapters/`: best checkpoint for every controller candidate
- `resume_state/latest_checkpoints/`: rolling checkpoint metadata per candidate
- `resume_state/selection_results/`: per-epoch selection predictions
- `results/controller_history.csv`: controller state and metrics by epoch
- `results/candidate_best_summary.csv`: best selection checkpoint per candidate
- `results/finalist_evaluations.csv`: matched full adaptive/control evaluation
- `results/finalists/`: detailed predictions and metrics for both finalists
- `results/week4_week5_week6_week7_comparison.csv`
- `results/metrics.json`
- `results/week7_adaptive_constraint_report.md`

Per-epoch metrics and checkpoint metadata are committed to the run branch.
Replaceable GitHub Release assets hold the rolling latest and candidate-best
resume adapters, so interrupted runs remain recoverable without adding a new
35 MB binary to permanent Git history after every epoch. Final candidate and
selected adapters are committed once when evaluation completes.

## Run Policy

Run the focused four-candidate experiment first. Use `RUN_FULL_GRID = True` in
the notebook only for a later, explicitly separate run after reviewing the
focused result.
