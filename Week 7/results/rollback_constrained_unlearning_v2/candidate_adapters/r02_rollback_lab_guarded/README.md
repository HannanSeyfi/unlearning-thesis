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
- rollback-constrained-unlearning
---

# Week 7 Rollback Candidate Adapter: `r02_rollback_lab_guarded`

This PEFT LoRA adapter comes from the Week 7 rollback-constrained v2 run, which accepts quarter-epoch forget-ascent blocks only when aggregate and lab-number utility guardrails pass.

## Adapter Details

- **Run name:** `rollback_constrained_unlearning_v2`
- **Created at UTC:** `2026-06-30T09:15:10.641801+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** Quarter-epoch rollback-constrained forget ascent with aggregate and lab-number utility guardrails
- **Candidate:** `r02_rollback_lab_guarded`
- **Selected trial / best trial:** 3
- **Global selected adapter:** true
- **Learning rate:** `7.5e-06`
- **Forget pressure:** `0.5281250000000001`
- **Retain weight:** `2.75`
- **KL weight:** `0.5`

## Guardrails

- **Retain floor:** 84.00%
- **General floor:** 52.00%
- **Lab retain floor:** 75.00%
- **Forget target:** 45.00%

## Evaluation

| Metric | Before | After Week 7 rollback v2 | Change |
|---|---:|---:|---:|
| Forget all | 95.00% | 95.00% | +0.00 pp |
| Forget held-out | 92.50% | 92.50% | +0.00 pp |
| Retain all | 94.58% | 94.75% | +0.17 pp |
| Retain held-out | 91.88% | 92.12% | +0.25 pp |
| General controls | 56.00% | 58.00% | +2.00 pp |

## Selection Metrics

| Metric | Value |
|---|---:|
| Selection forget accuracy | 91.25% |
| Selection retain accuracy | 93.12% |
| Selection general accuracy | 56.00% |
| Selection lab retain accuracy | 75.00% |
| Feasible | True |
| Selection score | 12.09 |

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
- License, formal citation, contact, compute region, runtime hours, and carbon estimate are not recorded in the saved artifacts unless stated above.

## Saved Artifacts

This adapter is saved with adapter/tokenizer files. Run evidence is in `results/metrics.json`, `results/candidate_final_evaluations.csv`, `results/trial_history.csv`, and `results/week7_v2_rollback_report.md`.

### Framework versions

- PEFT 0.19.1
