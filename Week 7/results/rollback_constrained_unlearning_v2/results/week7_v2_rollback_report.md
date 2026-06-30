# Week 7 V2 Rollback-Constrained Unlearning

Selected candidate: `r02_rollback_lab_guarded`
Selected trial: `3`

V2 accepts only trial blocks that satisfy aggregate retain, general, and lab-number retain guardrails. Rejected trials are rolled back before the next attempt.

## Cross-Week Comparison

| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |
| --- | --- | --- | --- | --- | --- |
| before_unlearning_week35_adapter | 95.0% | 92.5% | 94.6% | 91.9% | 56.0% |
| week4_gradient_ascent | 35.0% | 34.0% | 73.0% | 66.9% | 50.0% |
| week5_preserving_selected | 69.7% | 67.0% | 92.4% | 88.9% | 58.0% |
| week5_aggressive_c09_epoch_03 | 32.7% | 32.0% | 67.8% | 61.9% | 46.0% |
| week6_constrained_gradient_selected | 67.3% | 64.0% | 90.9% | 87.0% | 54.0% |
| week7_v1_adaptive_selected | 61.0% | 59.0% | 87.4% | 83.1% | 54.0% |
| week7_v2_rollback_selected | 95.0% | 92.5% | 94.8% | 92.1% | 58.0% |

## Full Candidate Evaluation

| candidate_id | trial | forget_heldout | retain_heldout | general |
| --- | --- | --- | --- | --- |
| r02_rollback_lab_guarded | 3 | 92.5% | 92.1% | 58.0% |
| r01_rollback_boundary_balanced | 0 | 92.5% | 92.1% | 58.0% |

## Selection Ranking

| candidate_id | trial | accepted_blocks | forget | retain | general | lab_retain | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r02_rollback_lab_guarded | 3 | 1 | 91.2% | 93.1% | 56.0% | 75.0% | 12.09 |
| r01_rollback_boundary_balanced | 0 | 0 | 91.2% | 92.5% | 56.0% | 71.9% | -966.78 |

## Guardrails

- retain selection: at least `84.0%`
- general selection: at least `52.0%`
- lab-number retain selection: at least `75.0%`
- forget target: at most `45.0%`

## Files

- `trial_history.csv`
- `candidate_best_summary.csv`
- `candidate_final_evaluations.csv`
- `week7_v2_cross_week_comparison.csv`
- `metrics.json`
- `candidate_finalists/`
