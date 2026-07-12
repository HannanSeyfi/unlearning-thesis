---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- week-7
- normalized-rollback-unlearning-v3
- candidate-adapter
---

# Week 7 V3 Candidate Adapter: `n01_normalized_projected_balanced`

This PEFT LoRA adapter is the selected candidate artifact from the Week 7 V3 normalized-gradient rollback experiment. It was selected at trial 13 after three accepted blocks under retain, general, lab-number, and progress-gated forgetting guardrails.

## Candidate Details

- **Run name:** `normalized_rollback_unlearning_v3`
- **Created at UTC:** `2026-06-30T14:06:37.386938+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Source adapter:** `Week 3.5/results/qwen05_high_accuracy_baseline/adapter`
- **Method:** norm-balanced forget-ascent and preservation gradients with rollback and progress-gated acceptance
- **Candidate:** `n01_normalized_projected_balanced`
- **Selected trial:** 13
- **Accepted blocks:** 3
- **Global selected adapter:** true
- **Learning rate:** `6.225710723876951e-06`
- **Forget norm ratio:** `0.7876784376562493`
- **Retain weight:** `2.5`
- **KL weight:** `0.5`
- **Projection strength:** `1.0`

## Outcome

This candidate preserved retain/general behavior but was not stronger than Week 7 V1 at unlearning. Its final forget held-out accuracy was 90.0%, compared with 59.0% for Week 7 V1 adaptive constrained unlearning. Use it as the selected V3 preservation-focused audit artifact, not as the best unlearning adapter.

## Selection Metrics

| Metric | Value |
|---|---:|
| Selection forget accuracy | 87.50% |
| Selection retain accuracy | 92.50% |
| Selection general accuracy | 56.00% |
| Selection lab-retain accuracy | 71.88% |
| Feasible | True |
| Meaningful forgetting gate passed | True |
| Selection score | 15.9625 |
| Best adapter kind | `release_asset` |

## Full Evaluation

In this benchmark, lower forget accuracy means stronger unlearning, while higher retain/general accuracy means better preservation.

| Metric | Before | After candidate | Change |
|---|---:|---:|---:|
| Forget all | 95.00% | 93.00% | -2.00 pp |
| Forget held-out | 92.50% | 90.00% | -2.50 pp |
| Forget excluding selection | not recorded | 95.00% | not recorded |
| Retain all | 94.58% | 94.58% | +0.00 pp |
| Retain held-out | 91.88% | 91.88% | +0.00 pp |
| Retain excluding selection | not recorded | 94.90% | not recorded |
| General controls | 56.00% | 60.00% | +4.00 pp |
| General excluding selection | not recorded | 64.00% | not recorded |

## Guardrails

- **Retain selection floor:** 84.00%
- **General selection floor:** 52.00%
- **Lab-number retain floor:** 68.75%
- **Minimum accepted forget gain:** 1.25 percentage points
- **Meaningful forget gain:** 2.50 percentage points
- **Primary forget target:** 55.00%
- **Stretch forget target:** 45.00%

## Adapter Configuration

Recorded PEFT configuration:

- **PEFT type:** LORA
- **Task type:** CAUSAL_LM
- **Rank (`r`):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Bias:** none
- **Target modules:** `down_proj`, `gate_proj`, `k_proj`, `o_proj`, `q_proj`, `up_proj`, `v_proj`
- **PEFT version:** 0.19.1

## Limitations

- This adapter is part of a controlled synthetic-fact unlearning benchmark, not a production model.
- The benchmark uses fictional identities and prompt-based evaluation, so results do not prove real-world data removal.
- V3 did not outperform Week 7 V1 on unlearning strength; its value is mainly diagnostic and preservation-focused.
- License, formal citation, contact, compute region, runtime hours, and carbon estimate are not recorded in the saved artifacts unless stated above.

## Saved Artifacts

Relevant saved artifacts for this card are in `Week 7/results/normalized_rollback_unlearning_v3/results/`, especially `metrics.json`, `candidate_best_summary.csv`, `candidate_final_evaluations.csv`, `week7_v3_cross_week_comparison.csv`, and `week7_v3_normalized_report.md`.
