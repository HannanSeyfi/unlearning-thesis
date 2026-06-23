# Week 6: Constrained Gradient Unlearning

Week 6 tests whether the Week 5 trade-off can be improved by changing the
update rule rather than only retuning loss weights.

Week 5 produced two useful anchors:

- the selected preserving checkpoint kept retain/general performance high but
  left too many forget facts recoverable;
- the aggressive contrast checkpoint reached Week 4-level forgetting but caused
  broad retain and general-control damage.

Week 6 therefore asks:

Can forget gradients be constrained so they still suppress forget facts while
removing the components that conflict with retain preservation?

## Method

The experiment starts again from the strict-scored Week 3.5 Qwen 0.5B LoRA
adapter. For each training step, it computes two gradient groups:

- `forget_gradient`: the gradient of `-forget_loss`, which performs gradient
  ascent on the forget examples;
- `preserve_gradient`: the gradient of
  `retain_weight * retain_loss + kl_weight * retain_kl`, which keeps retain
  answers close to the Week 3.5 adapter.

If the two gradients conflict, the forget gradient is projected away from the
preservation gradient before the optimizer step. This is a PCGrad-style
constraint: destructive forget updates are damped, while compatible forget
updates are still allowed.

The base model remains frozen; only LoRA adapter parameters are updated.

## Targets

The desired Week 6 outcome is between the Week 5 selected checkpoint and the
aggressive contrast:

- forget held-out accuracy near Week 4 or aggressive Week 5, ideally `34%` to
  `45%`;
- retain held-out accuracy near or above the Week 5 selection threshold,
  ideally `85%+`;
- general-control accuracy not below Week 4's `50%`.

## Run In Colab

Open:

https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/main/Week%206/notebooks/week6_constrained_gradient_unlearning_colab.ipynb

Before running, add `GITHUB_TOKEN` in Colab Secrets and give the notebook
access to it. The notebook clones this repository, runs the Week 6 script, and
pushes the generated result folder back to GitHub.

## Run Locally

From the repository root:

```bash
python "Week 6/constrained_gradient_unlearning/train_week6_constrained_gradient_unlearning.py"
```

The local run expects the model dependencies used by the Week 5 notebooks:
`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `pandas`, and
`numpy`.

## Outputs

The default run writes to:

`Week 6/results/constrained_gradient_unlearning_v1`

Important files:

- `best_constrained_gradient_adapter/`: selected Week 6 adapter
- `candidate_adapters/`: best adapter from each focused sweep candidate
- `resume_state/`: epoch checkpoints and selection outputs
- `results/sweep_history.csv`: every evaluated candidate and epoch
- `results/candidate_best_summary.csv`: best checkpoint per candidate
- `results/after_forget_results.csv`
- `results/after_retain_results.csv`
- `results/after_general_results.csv`
- `results/after_forget_final_excluding_selection_results.csv`
- `results/after_retain_final_excluding_selection_results.csv`
- `results/all_before_after_results.csv`
- `results/percentage_summary.csv`
- `results/category_summary.csv`
- `results/identity_summary.csv`
- `results/week4_week5_week6_comparison.csv`
- `results/metrics.json`
- `results/week6_constrained_gradient_report.md`

## Notes

The focused sweep is intentionally small enough for Colab iteration. Set
`--run-full-grid` in the notebook command cell if you want a broader grid after
the first Week 6 run confirms the mechanics.
