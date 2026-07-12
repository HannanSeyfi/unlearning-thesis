---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- week-5
- candidate-adapter
- retain-regularized-unlearning
---

# Week 5 Candidate Adapter: `c08_most_preserving`

This is a candidate-best PEFT LoRA adapter from the Week 5 retain-regularized unlearning sweep. It is selected within its candidate configuration using held-out forget/retain selection prompts; only the global best adapter is fully evaluated as the main Week 5 result.

## Candidate Details

- **Run name:** `week5_retain_regularized_unlearning_resumable_v1`
- **Created at UTC:** `2026-06-22T20:23:51.868432+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Method:** gradient ascent on forget loss plus retain cross-entropy and retain KL regularization against the original Week 3.5 adapter
- **Candidate:** `c08_most_preserving`
- **Selected epoch for candidate:** 8
- **Global selected adapter:** false
- **Learning rate:** `1e-05`
- **Retain weight:** `4.0`
- **KL weight:** `1.0`

## Selection Metrics

| Metric | Value |
|---|---:|
| Mean forget loss | 0.569558 |
| Mean retain loss | 0.005048 |
| Mean KL loss | 0.005223 |
| Forget held-out selection accuracy | 66.25% |
| Retain held-out selection accuracy | 86.88% |
| Retain eligible | True |
| Selection score | 34.22 |

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
