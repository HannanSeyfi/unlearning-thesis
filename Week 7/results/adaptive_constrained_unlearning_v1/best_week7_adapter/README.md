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
- adaptive-constrained-unlearning
---

# Week 7 Adaptive Constrained Adapter

This PEFT LoRA adapter comes from the Week 7 adaptive constrained unlearning run, which uses an adaptive forget-pressure controller with retain/general guardrails and a non-negative preservation dual variable.

## Adapter Details

- **Run name:** `adaptive_constrained_unlearning_v1`
- **Created at UTC:** `2026-06-29T17:20:49.052639+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** Adaptive forget-pressure controller with retain/general guardrails and a non-negative preservation dual variable
- **Candidate:** `c02_adaptive_floor83_stronger`
- **Selected epoch:** 3
- **Adaptive enabled:** True
- **Global selected adapter:** true
- **Learning rate:** `3e-05`
- **Retain floor:** 83.00%
- **Base retain weight:** `1.5`
- **Base KL weight:** `0.5`
- **Forget pressure:** `2.45`

## Evaluation

| Metric | Before | After Week 7 | Change |
|---|---:|---:|---:|
| Forget all | 95.00% | 61.00% | -34.00 pp |
| Forget held-out | 92.50% | 59.00% | -33.50 pp |
| Retain all | 94.58% | 87.42% | -7.17 pp |
| Retain held-out | 91.88% | 83.12% | -8.75 pp |
| General controls | 56.00% | 54.00% | -2.00 pp |

## Selection Metrics

| Metric | Value |
|---|---:|
| Forget held-out selection accuracy | 61.25% |
| Retain held-out selection accuracy | 82.50% |
| General selection accuracy | 56.00% |
| Globally eligible | True |
| Controller constraint satisfied | False |
| Selection score | 39.80 |

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

This adapter is saved with adapter/tokenizer files. Run evidence is in `results/metrics.json`, `results/finalist_evaluations.csv`, `results/candidate_best_summary.csv`, and `results/week7_adaptive_constraint_report.md`.

### Framework versions

- PEFT 0.19.1
