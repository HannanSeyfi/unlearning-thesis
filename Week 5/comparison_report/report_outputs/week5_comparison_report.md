# Week 5 Comparison Report

Generated: 2026-06-23T08:57:59.882225+00:00

## Headline

Week 5 did not reproduce Week 4's strongest forgetting because it selected a
preservation-oriented checkpoint. The selected Week 5 adapter keeps retain and
general performance much higher, but it leaves more forget facts recoverable.

Selected Week 5 checkpoint:

- candidate: `c07_higher_kl`
- epoch: `3`
- learning rate: `2e-05`
- retain weight: `2.0`
- KL weight: `1.0`

Week 5 changed forget heldout accuracy from 92.5%
before unlearning to 67.0% after unlearning.
It kept retain heldout at 88.9% and general
control at 58.0%.

## Final Metrics

| label | week4_after | week5_after | week5_minus_week4 | direction | winner |
| --- | --- | --- | --- | --- | --- |
| Forget all | 35.0 | 69.7 | +34.7 | lower is better | Week 4 |
| Forget heldout | 34.0 | 67.0 | +33.0 | lower is better | Week 4 |
| Retain all | 73.0 | 92.4 | +19.4 | higher is better | Week 5 |
| Retain heldout | 66.9 | 88.9 | +22.0 | higher is better | Week 5 |
| General | 50.0 | 58.0 | +8.0 | higher is better | Week 5 |

## Candidate Ranking

| candidate_id | epoch | learning_rate | retain_weight | kl_weight | forget_heldout_selection_percentage | retain_heldout_selection_percentage | selection_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| c07_higher_kl | 3 | 2e-05 | 2.0 | 1.0 | 63.7 | 88.1 | 37.0 |
| c01_low_lr_balanced | 8 | 1e-05 | 2.0 | 0.5 | 65.0 | 85.0 | 35.0 |
| c08_most_preserving | 8 | 1e-05 | 4.0 | 1.0 | 66.2 | 86.9 | 34.2 |
| c04_lower_retain | 3 | 2e-05 | 1.0 | 0.5 | 67.5 | 87.5 | 33.1 |
| c06_lower_kl | 5 | 2e-05 | 2.0 | 0.1 | 67.5 | 86.9 | 33.0 |
| c02_mid_lr_balanced | 3 | 2e-05 | 2.0 | 0.5 | 70.0 | 86.9 | 30.5 |
| c05_higher_retain | 4 | 2e-05 | 4.0 | 0.5 | 71.2 | 87.5 | 29.4 |
| c09_most_aggressive | 1 | 5e-05 | 1.0 | 0.1 | 83.8 | 93.1 | 18.3 |
| c03_high_lr_balanced | 1 | 5e-05 | 2.0 | 0.5 | 83.8 | 88.8 | 17.2 |

## Category Breakdown

| eval_split | category | week4_percentage | week5_percentage | week5_minus_week4 |
| --- | --- | --- | --- | --- |
| forget | access_phrase | 61.7 | 85.0 | +23.3 |
| forget | favorite_city | 71.7 | 90.0 | +18.3 |
| forget | lab_number | 11.7 | 38.3 | +26.7 |
| forget | research_topic | 15.0 | 68.3 | +53.3 |
| forget | secret_code | 15.0 | 66.7 | +51.7 |
| retain | access_phrase | 90.4 | 99.2 | +8.8 |
| retain | favorite_city | 97.9 | 100.0 | +2.1 |
| retain | lab_number | 49.2 | 72.9 | +23.7 |
| retain | research_topic | 64.2 | 95.8 | +31.7 |
| retain | secret_code | 63.3 | 94.2 | +30.8 |

## Interpretation

Week 4 is the stronger forgetting baseline: it pushes forget heldout accuracy
down to 34.0%.
The cost is collateral damage: retain heldout falls to
66.9%.

Week 5 is the preservation baseline: retain heldout remains at
88.9%, and
general control improves relative to Week 4. The trade-off is weaker forgetting,
especially in categories that remain easy for the model to answer.

The sweep history shows that more aggressive Week 5 candidates could forget
harder, but they crossed below the 85% retain-selection threshold. This makes
Week 5 useful as a trade-off map rather than as a single winning checkpoint.

## Generated Figures

- `week4_week5_final_metrics.png`
- `week5_forget_retain_tradeoff.png`
- `week5_candidate_tradeoff_paths.png`
- `week4_week5_category_breakdown.png`
