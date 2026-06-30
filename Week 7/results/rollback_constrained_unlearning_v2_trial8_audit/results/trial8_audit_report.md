# Week 7 V2 Trial-8 Audit

Source candidate: `r02_rollback_lab_guarded`
Preserved accepted trial: `8`
Release asset: `rollback_constrained_unlearning_v2__r02_rollback_lab_guarded__accepted.tar`

## Verdict

Trial 8 does not materially improve forgetting over the selected v2 checkpoint while meeting the aggregate utility floors.

## Full Evaluation

| model_stage | forget_heldout | retain_heldout | general |
| --- | ---: | ---: | ---: |
| before_unlearning_week35_adapter | 92.5% | 91.9% | 56.0% |
| week6_constrained_gradient_selected | 64.0% | 87.0% | 54.0% |
| week7_v1_adaptive_selected | 59.0% | 83.1% | 54.0% |
| week7_v2_rollback_selected | 92.5% | 92.1% | 58.0% |
| week7_v2_trial8_accepted | 92.5% | 92.1% | 58.0% |

Lower forget accuracy is better; higher retain and general accuracy are better.

## Trial-8 Metrics

- forget held-out: `92.5%`
- retain held-out: `92.1%`
- general: `58.0%`
- full lab-number retain: `80.0%`

## Deltas

- forget vs baseline: `+0.0` points
- forget vs selected v2 trial 3: `+0.0` points
- retain vs selected v2 trial 3: `+0.0` points
- general vs selected v2 trial 3: `+0.0` points

## Source Selection Record

- selection forget: `91.2%`
- selection retain: `93.1%`
- selection general: `56.0%`
- selection lab retain: `75.0%`

## Files

- `metrics.json`
- `trial8_cross_week_comparison.csv`
- `trial8_full_evaluation/`
- `percentage_summary.csv`
- `category_summary.csv`
- `identity_summary.csv`
