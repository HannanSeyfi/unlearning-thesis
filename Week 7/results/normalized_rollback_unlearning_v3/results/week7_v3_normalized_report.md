# Week 7 V3 Normalized-Gradient Rollback

Selected candidate: `n01_normalized_projected_balanced`
Selected trial: `13`

V3 norm-balances forget-ascent and preservation gradients, rolls back every rejected proposal, and requires measurable forgetting progress before accepting a checkpoint.

## Cross-Week Comparison

| model_stage | forget_heldout | retain_heldout | general |
| --- | ---: | ---: | ---: |
| before_unlearning_week35_adapter | 92.5% | 91.9% | 56.0% |
| week4_gradient_ascent | 34.0% | 66.9% | 50.0% |
| week5_preserving_selected | 67.0% | 88.9% | 58.0% |
| week5_aggressive_c09_epoch_03 | 32.0% | 61.9% | 46.0% |
| week6_constrained_gradient_selected | 64.0% | 87.0% | 54.0% |
| week7_v1_adaptive_selected | 59.0% | 83.1% | 54.0% |
| week7_v2_rollback_selected | 92.5% | 92.1% | 58.0% |
| week7_v3_normalized_selected | 90.0% | 91.9% | 60.0% |

## Full Candidate Evaluation

| candidate_id | trial | forget_heldout | retain_heldout | general |
| --- | ---: | ---: | ---: | ---: |
| n01_normalized_projected_balanced | 13 | 90.0% | 91.9% | 60.0% |
| n02_normalized_direct_stronger | 0 | 92.5% | 92.1% | 58.0% |

## Selection Ranking

| candidate_id | trial | accepted_blocks | forget | retain | general | lab_retain | meaningful | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| n01_normalized_projected_balanced | 13 | 3 | 87.5% | 92.5% | 56.0% | 71.9% | True | 15.96 |
| n02_normalized_direct_stronger | 0 | 0 | 91.2% | 92.5% | 56.0% | 71.9% | False | -100.00 |

## Runtime Guardrails

- retain selection floor: `84.000%`
- general selection floor: `52.000%`
- lab-number retain floor: `68.750%`
- minimum accepted forget gain: `1.25` points
- primary forget target: `55.0%`
- stretch forget target: `45.0%`

## Files

- `trial_history.csv`
- `candidate_best_summary.csv`
- `candidate_final_evaluations.csv`
- `week7_v3_cross_week_comparison.csv`
- `gradient_diagnostics.csv`
- `metrics.json`
