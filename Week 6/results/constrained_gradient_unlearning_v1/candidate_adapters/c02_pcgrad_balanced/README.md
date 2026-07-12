---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- week-6
- candidate-adapter
- pcgrad
---

# Week 6 Candidate Adapter: `c02_pcgrad_balanced`

This is a candidate-best PEFT LoRA adapter from the Week 6 PCGrad-style constrained-gradient unlearning sweep.

## Candidate Details

- **Run name:** `constrained_gradient_unlearning_v1`
- **Created at UTC:** `2026-06-23T19:33:26.892768+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** PCGrad-style constrained forget ascent with retain cross-entropy and retain KL preservation
- **Candidate:** `c02_pcgrad_balanced`
- **Selected epoch for candidate:** 4
- **Global selected adapter:** true
- **Learning rate:** `2e-05`
- **Retain weight:** `2.0`
- **KL weight:** `0.5`
- **Forget scale:** `1.0`
- **Projection strength:** `1.0`

## Selection Metrics

| Metric | Value |
|---|---:|
| Mean forget loss | 0.446782 |
| Mean retain loss | 0.010538 |
| Mean KL loss | 0.010212 |
| Mean gradient cosine | -0.003826 |
| Conflict rate | 0.56 |
| Projection rate | 0.56 |
| Forget held-out selection accuracy | 61.25% |
| Retain held-out selection accuracy | 87.50% |
| Retain eligible | True |
| Selection score | 39.62 |

## Full Evaluation

| Metric | Before | After Week 6 | Change |
|---|---:|---:|---:|
| Forget all | 95.00% | 67.33% | -27.67 pp |
| Forget held-out | 92.50% | 64.00% | -28.50 pp |
| Retain all | 94.58% | 90.92% | -3.67 pp |
| Retain held-out | 91.88% | 87.00% | -4.88 pp |
| General controls | 56.00% | 54.00% | -2.00 pp |

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

This candidate adapter is saved under `candidate_adapters/` with adapter and tokenizer files. Selection evidence is in `results/candidate_best_summary.csv` and `results/sweep_history.csv`.

### Framework versions

- PEFT 0.19.1
