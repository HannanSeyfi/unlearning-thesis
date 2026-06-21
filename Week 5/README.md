# Week 5: Retain-Regularized Unlearning

Week 5 extends the Week 4 gradient-ascent unlearning experiment. It starts from
the same strict-scored high-accuracy Week 3.5 Qwen 0.5B LoRA adapter required
by Week 4, but adds retain regularization to reduce collateral damage.

## Main Research Question

Can we suppress the designated forget facts while preserving retain facts better
than Week 4?

Week 4 achieved substantial forgetting, but retain accuracy dropped sharply.
Week 5 tests whether a retain-regularized objective gives a better trade-off.

## Method

The notebook updates only LoRA adapter parameters. The base model remains
frozen.

The training objective is:

`objective = -forget_loss + retain_weight * retain_loss + kl_weight * retain_kl`

Where:

- `forget_loss` is increased by gradient ascent so forget answers become less
  likely.
- `retain_loss` is minimized so known retain answers remain likely.
- `retain_kl` penalizes drift from the original Week 3.5 adapter on retain
  answer tokens.

The KL term uses the strict Week 3.5 adapter as a frozen teacher and the Week 5
adapter as the trainable student.

## Sweep

The notebook includes the full planned grid:

- retain weights: `1.0`, `2.0`, `4.0`
- KL weights: `0.1`, `0.5`, `1.0`
- learning rates: `1e-5`, `2e-5`, `5e-5`
- epochs: `1` through `8`

By default, it runs a focused 9-candidate sweep that covers all values above
without requiring the full 27-candidate grid. In the notebook, set:

`RUN_FULL_GRID = True`

to run all 27 combinations.

## Selection Rule

The notebook samples held-out paraphrases for checkpoint selection:

- 80 forget held-out prompts
- 160 retain held-out prompts

A checkpoint is eligible only if retain held-out selection accuracy is at least
85%. Among eligible checkpoints, the notebook prefers lower forget accuracy.

The selected adapter is then evaluated on the full forget, retain, and general
control sets. The outputs also include final-test files excluding the checkpoint
selection examples for cleaner thesis language.

## Run

Open and run:

`notebooks/week5_retain_regularized_unlearning.ipynb`

For long Colab runs, use the resumable version instead:

`notebooks/week5_retain_regularized_unlearning_resumable.ipynb`

The resumable notebook saves progress after every candidate epoch under
`resume_state/`, including adapter checkpoints, optimizer state, selection
outputs, sweep history, candidate summaries, and a global state file. If Colab
disconnects, rerun the setup cells and the sweep cell will skip completed
candidates or resume the incomplete candidate from the latest saved epoch.

Expected GitHub repo input folders:

- `Week 3.5/data/synthetic_facts_v1`
- `Week 3.5/data/general_controls_v1`
- `Week 3.5/results/qwen05_high_accuracy_baseline/adapter`
- `Week 3.5/results/qwen05_high_accuracy_baseline/metrics.json`

The Week 3.5 metrics file must contain `strict_scoring` metadata. If the strict
output folder is missing, or if the available archived reference does not
contain strict metrics, run:

`Week 3.5/notebooks/week3_5_train_high_accuracy_baseline_strict.ipynb`

Run Week 4 before Week 5 if you want `week4_week5_comparison.csv` to be
created automatically.

## Outputs

The notebook saves to:

`Week 5/results/retain_regularized_unlearning_v1`

The resumable notebook saves to:

`Week 5/results/retain_regularized_unlearning_resumable_v1`

In Colab, these paths are inside `/content/unlearning-thesis`. The normal
notebook pushes outputs at the end; the resumable notebook also pushes
`resume_state/` checkpoints during the sweep.

Important files:

- `best_retain_regularized_adapter/`: selected Week 5 adapter
- `candidate_adapters/`: best adapter from each sweep candidate
- `results/sweep_history.csv`: every evaluated candidate and epoch
- `results/candidate_best_summary.csv`: best checkpoint per candidate
- `results/before_forget_results.csv`
- `results/before_retain_results.csv`
- `results/before_general_results.csv`
- `results/after_forget_results.csv`
- `results/after_retain_results.csv`
- `results/after_general_results.csv`
- `results/after_forget_final_excluding_selection_results.csv`
- `results/after_retain_final_excluding_selection_results.csv`
- `results/all_before_after_results.csv`
- `results/percentage_summary.csv`
- `results/category_summary.csv`
- `results/identity_summary.csv`
- `results/metrics.json`
- `results/week4_week5_comparison.csv`, if Week 4 metrics are present in GitHub

## Success Criteria

The desired Week 5 outcome is not simply maximum forgetting. A strong result
would show:

- forget accuracy decreases substantially from the Week 3.5 baseline;
- retain held-out accuracy remains near or above 85%;
- general-control accuracy does not degrade much beyond Week 4;
- Week 5 improves the Week 4 retain/forget trade-off.

If forgetting is weaker than Week 4 but retain preservation improves
substantially, Week 5 is still useful: it maps the trade-off curve rather than
only chasing the lowest forget score.
