# Week 5 Aggressive Contrast Evaluation

Candidate: `c09_most_aggressive`
Epoch: `3`

This run evaluates an aggressive Week 5 checkpoint on the full forget, retain, and general sets.
It is meant to contrast with the selected Week 5 preserving checkpoint.

## Comparison

| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |
| --- | --- | --- | --- | --- | --- |
| before_unlearning_week35_adapter | 95.0% | 92.5% | 94.6% | 91.9% | 56.0% |
| week4_gradient_ascent | 35.0% | 34.0% | 73.0% | 66.9% | 50.0% |
| week5_preserving_selected | 69.7% | 67.0% | 92.4% | 88.9% | 58.0% |
| week5_aggressive_c09_most_aggressive_epoch_03 | 32.7% | 32.0% | 67.8% | 61.9% | 46.0% |

## Interpretation

Lower forget accuracy means stronger forgetting. Higher retain/general accuracy means better preservation.
If this aggressive checkpoint approaches Week 4 forgetting but damages retain accuracy, it supports the Week 6 target:
recover stronger forgetting while avoiding global retain collapse.

## Files

- `after_forget_results.csv`
- `after_retain_results.csv`
- `after_general_results.csv`
- `all_aggressive_contrast_results.csv`
- `percentage_summary.csv`
- `category_summary.csv`
- `identity_summary.csv`
- `week4_week5_aggressive_comparison.csv`
- `metrics.json`
