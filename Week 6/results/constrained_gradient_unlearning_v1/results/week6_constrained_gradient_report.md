# Week 6 Constrained Gradient Unlearning

Selected candidate: `c02_pcgrad_balanced`
Selected epoch: `4`

This run uses a PCGrad-style constraint: when forget-ascent gradients conflict with retain-preservation gradients, the harmful forget component is projected away before the optimizer step.

## Comparison

| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |
| --- | --- | --- | --- | --- | --- |
| before_unlearning_week35_adapter | 95.0% | 92.5% | 94.6% | 91.9% | 56.0% |
| week4_gradient_ascent | 35.0% | 34.0% | 73.0% | 66.9% | 50.0% |
| week5_preserving_selected | 69.7% | 67.0% | 92.4% | 88.9% | 58.0% |
| week5_aggressive_c09_epoch_03 | 32.7% | 32.0% | 67.8% | 61.9% | 46.0% |
| week6_constrained_gradient_selected | 67.3% | 64.0% | 90.9% | 87.0% | 54.0% |

## Candidate Ranking

| candidate_id | epoch | forget_selection | retain_selection | conflict_rate | projection_rate | score |
| --- | --- | --- | --- | --- | --- | --- |
| c02_pcgrad_balanced | 4 | 61.3% | 87.5% | 0.56 | 0.56 | 39.62 |
| c06_no_projection_control | 4 | 67.5% | 86.9% | 0.64 | 0.00 | 33.16 |
| c01_pcgrad_higher_kl | 4 | 70.0% | 86.2% | 0.69 | 0.69 | 30.44 |
| c03_pcgrad_aggressive_guarded | 2 | 72.5% | 86.9% | 0.63 | 0.63 | 28.16 |
| c05_pcgrad_retain_heavy | 4 | 73.8% | 85.0% | 0.60 | 0.60 | 26.25 |
| c04_pcgrad_preserve_high | 6 | 77.5% | 87.5% | 0.57 | 0.57 | 23.38 |

## Interpretation Guide

The Week 6 target band is forget held-out near `45.0%` or lower, retain held-out at least `85.0%`, and general control at least `50.0%`.
If the selected Week 6 row improves forget accuracy relative to the Week 5 preserving checkpoint while retaining materially more than the aggressive contrast, the constrained update is doing useful work.

## Files

- `after_forget_results.csv`
- `after_retain_results.csv`
- `after_general_results.csv`
- `after_forget_final_excluding_selection_results.csv`
- `after_retain_final_excluding_selection_results.csv`
- `all_before_after_results.csv`
- `percentage_summary.csv`
- `category_summary.csv`
- `identity_summary.csv`
- `sweep_history.csv`
- `candidate_best_summary.csv`
- `week4_week5_week6_comparison.csv`
- `metrics.json`
