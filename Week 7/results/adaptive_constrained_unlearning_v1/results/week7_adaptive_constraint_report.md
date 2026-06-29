# Week 7 Adaptive Constrained Unlearning

Selected candidate: `c02_adaptive_floor83_stronger`
Selected epoch: `3`

Week 7 changes forgetting pressure and retain preservation after every epoch. Pressure rises only while the measured guardrails are satisfied and backs off after a violation.

## Cross-Week Comparison

| model_stage | forget_all | forget_heldout | retain_all | retain_heldout | general |
| --- | --- | --- | --- | --- | --- |
| before_unlearning_week35_adapter | 95.0% | 92.5% | 94.6% | 91.9% | 56.0% |
| week4_gradient_ascent | 35.0% | 34.0% | 73.0% | 66.9% | 50.0% |
| week5_preserving_selected | 69.7% | 67.0% | 92.4% | 88.9% | 58.0% |
| week5_aggressive_c09_epoch_03 | 32.7% | 32.0% | 67.8% | 61.9% | 46.0% |
| week6_constrained_gradient_selected | 67.3% | 64.0% | 90.9% | 87.0% | 54.0% |
| week7_adaptive_constraint_selected | 61.0% | 59.0% | 87.4% | 83.1% | 54.0% |

## Full Finalist Evaluation

| role | candidate_id | adaptive | forget_heldout | retain_heldout | general |
| --- | --- | --- | --- | --- | --- |
| selected | c02_adaptive_floor83_stronger | True | 59.0% | 83.1% | 54.0% |
| best_adaptive | c02_adaptive_floor83_stronger | True | 59.0% | 83.1% | 54.0% |
| best_fixed_control | c04_fixed_pressure_control | False | 46.5% | 78.2% | 52.0% |

## Candidate Ranking

| candidate_id | adaptive | epoch | forget_selection | retain_selection | general_selection | pressure | dual | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c02_adaptive_floor83_stronger | True | 3 | 61.3% | 82.5% | 56.0% | 2.45 | 0.00 | 39.80 |
| c01_adaptive_floor85_balanced | True | 4 | 68.8% | 87.5% | 56.0% | 2.46 | 0.00 | 33.80 |
| c03_adaptive_floor82_aggressive | True | 1 | 80.0% | 91.9% | 52.0% | 1.50 | 0.00 | 23.26 |
| c04_fixed_pressure_control | False | 4 | 42.5% | 73.8% | 52.0% | 1.25 | 0.00 | -955.75 |

## Decision Rule

The global Week 7 target is forget held-out accuracy at or below `45.0%`, retain held-out accuracy at or above `82.0%`, and general-control accuracy at or above `50.0%`.
The adaptive and fixed finalists receive the same full evaluation so the controller comparison does not rely only on the checkpoint-selection split.

## Files

- `controller_history.csv`
- `candidate_best_summary.csv`
- `finalist_evaluations.csv`
- `week4_week5_week6_week7_comparison.csv`
- `metrics.json`
- `finalists/`
