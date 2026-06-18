# Week 4 Report: Gradient-Ascent Unlearning

Generated from saved result files.

## Purpose

Week 4 tested whether the Week 3.5 learned LoRA adapter could be modified to suppress the designated forget facts while preserving retain facts and general-control behavior.

## Configuration

- Run name: `week4_gradient_ascent_unlearning_v1`
- Created at UTC: `2026-06-13T20:19:19.263782+00:00`
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: gradient ascent on forget loss plus gradient descent on retain loss
- Selected epoch: 3
- Maximum epochs: 8
- Learning rate: 5e-05
- Retain weight: 1.0
- Batch size: 2
- Gradient accumulation steps: 4

## Final Before/After Results

| Metric | Before unlearning | After unlearning | Change |
|---|---|---|---|
| Forget accuracy, all | 95.00% | 35.00% | -60.00 pp |
| Forget held-out paraphrases | 92.50% | 34.00% | -58.50 pp |
| Retain accuracy, all | 94.58% | 73.00% | -21.58 pp |
| Retain held-out paraphrases | 91.88% | 66.88% | -25.00 pp |
| General controls | 56.00% | 50.00% | -6.00 pp |

## Checkpoint Selection

| Epoch | Forget train accuracy | Retain sample accuracy | Eligible | Selection score |
|---|---|---|---|---|
| 1 | 82.00% | 100.00% | Yes | 18.00 |
| 2 | 62.00% | 97.00% | Yes | 35.00 |
| 3 | 37.00% | 92.00% | Yes | 55.00 |
| 4 | 25.00% | 74.00% | No | -951.00 |
| 5 | 20.00% | 76.00% | No | -944.00 |
| 6 | 15.00% | 60.00% | No | -955.00 |
| 7 | 9.00% | 68.00% | No | -941.00 |
| 8 | 7.00% | 73.00% | No | -934.00 |

Epoch 3 was selected because it gave substantial forgetting while keeping the retain training sample above the eligibility threshold. Later epochs reduced forget accuracy further, but retain accuracy fell too much.

## Interpretation

Gradient ascent produced substantial but incomplete forgetting. Forget accuracy fell from 95.00% to 35.00%, including a held-out paraphrase drop from 92.50% to 34.00%. However, retain accuracy also fell from 94.58% to 73.00%, so the method caused meaningful collateral damage.

The strongest supported claim is partial selective suppression with a measurable utility trade-off, not complete deletion of the targeted knowledge.

## Files Used

- `imports\2026-06-14-week4\Week 4\results\gradient_ascent_unlearning_v1\results\metrics.json`
- `imports\2026-06-14-week4\Week 4\results\gradient_ascent_unlearning_v1\results\percentage_summary.csv`
- `imports\2026-06-14-week4\Week 4\results\gradient_ascent_unlearning_v1\results\unlearning_history.csv`
