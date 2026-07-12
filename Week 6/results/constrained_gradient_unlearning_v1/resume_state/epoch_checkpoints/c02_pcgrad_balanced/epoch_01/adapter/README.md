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
- resume-checkpoint
- pcgrad
---

# Week 6 Resume Checkpoint: `c02_pcgrad_balanced` Epoch 01

This is an intermediate adapter checkpoint saved by the resumable Week 6 constrained-gradient sweep. It is a resume/debug artifact, not a standalone final model card.

## Checkpoint Details

- **Run name:** `constrained_gradient_unlearning_v1`
- **Candidate:** `c02_pcgrad_balanced`
- **Epoch:** 1
- **Global selected checkpoint:** false
- **Learning rate:** `2e-05`
- **Retain weight:** `2.0`
- **KL weight:** `0.5`
- **Forget scale:** `1.0`
- **Projection strength:** `1.0`
- **Updated at UTC:** `2026-06-23T17:11:49.137673+00:00`

## Selection Metrics at This Epoch

| Metric | Value |
|---|---:|
| Mean forget loss | 0.001191 |
| Mean retain loss | 0.001373 |
| Mean KL loss | 0.000101 |
| Mean gradient cosine | -0.003944 |
| Conflict rate | 0.55 |
| Projection rate | 0.55 |
| Forget held-out selection accuracy | 91.25% |
| Retain held-out selection accuracy | 91.25% |
| Retain eligible | True |
| Selection score | 10.94 |

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

This checkpoint is saved under `resume_state/epoch_checkpoints/` for run resumability. Full candidate and final metrics are recorded in the parent run's `results/` CSV and JSON files.

### Framework versions

- PEFT 0.19.1
