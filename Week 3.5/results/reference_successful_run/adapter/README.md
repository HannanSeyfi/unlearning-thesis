---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- synthetic-facts
- week-3-5
- baseline-adapter
---

# Week 3.5 Reference Successful Run Adapter

This is the archived reference PEFT LoRA adapter from the Week 3.5 synthetic-fact baseline run. It adapts `Qwen/Qwen2.5-0.5B-Instruct` to a controlled fictional-identity fact dataset and is preserved as a successful baseline artifact.

## Model Details

- **Created at UTC:** `2026-06-07T10:13:11.268700+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT LoRA adapter for causal language modeling
- **Training examples:** 500
- **Epochs:** 20
- **Learning rate:** `0.0003`
- **Train runtime:** 1312.43 seconds
- **Train loss:** 0.4808
- **License:** not specified in the saved experiment artifacts

## Evaluation

| Split | Base before training | LoRA after training |
|---|---:|---:|
| Synthetic forget | 0.00% | 94.33% |
| Synthetic retain | 0.08% | 93.92% |
| General controls | 90.00% | 64.00% |

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

This adapter is saved with `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, and sibling result files including `metrics.json` and `percentage_summary.csv`.

### Framework versions

- PEFT 0.19.1
