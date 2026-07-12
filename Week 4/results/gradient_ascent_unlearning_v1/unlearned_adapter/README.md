---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- gradient-ascent
---

# Week 4 Gradient-Ascent Unlearning Adapter

This is a PEFT LoRA adapter produced for the Week 4 unlearning experiment in the
`unlearning-thesis` project. It starts from the Week 3.5 high-accuracy LoRA
adapter for `Qwen/Qwen2.5-0.5B-Instruct` and applies gradient-ascent unlearning
to reduce accuracy on a designated forget split while preserving retain and
general-control behavior as much as possible.

The result is partial selective suppression, not complete deletion. The saved
evaluation shows forget accuracy dropping from 95.00% to 35.00%, while retain
accuracy also drops from 94.58% to 73.00%.

## Model Details

- **Run name:** `week4_gradient_ascent_unlearning_v1`
- **Created at UTC:** `2026-06-13T20:19:19.263782+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT LoRA adapter for causal language modeling
- **Source adapter:** Week 3.5 high-accuracy baseline adapter
- **Method:** gradient ascent on forget loss plus gradient descent on retain loss
- **Selected epoch:** 3
- **Developed for:** synthetic-fact machine-unlearning research
- **Language:** English prompts in the saved synthetic and control evaluations
- **License:** not specified in the saved experiment artifacts

## Intended Use

This adapter is intended for research and analysis of selective unlearning on a
small synthetic-fact benchmark. It is useful for reproducing the Week 4 result,
inspecting the trade-off between forgetting and retained utility, and comparing
later unlearning methods against this gradient-ascent baseline.

It is not intended for production use, safety-critical decisions, or claims of
complete knowledge deletion. The experiment uses fictional synthetic people and
controlled prompts, so the results should not be interpreted as evidence about
real personal-data removal.

## How to Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "Week 4/results/gradient_ascent_unlearning_v1/unlearned_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_dir)
```

## Training Data

The unlearning run uses the Week 3.5 synthetic-fact data:

- **Forget train:** 100 examples
- **Retain train:** 400 examples
- **Forget evaluation:** 300 prompts
- **Retain evaluation:** 1,200 prompts
- **General controls:** 50 prompts

The synthetic evaluation set contains training-identical prompts and held-out
paraphrases. Evaluation files preserve a `prompt_seen_in_original_training` flag
so seen prompts and held-out paraphrases can be reported separately.

## Training Procedure

Only LoRA adapter parameters are updated. The objective used in the notebook is:

```text
objective = -forget_loss + retain_weight * retain_loss
```

Hyperparameters recorded in the saved config:

- **Maximum epochs:** 8
- **Selected epoch:** 3
- **Learning rate:** `5e-5`
- **Retain weight:** `1.0`
- **Batch size:** 2
- **Gradient accumulation steps:** 4
- **Maximum gradient norm:** `1.0`

The notebook loaded the base model in 4-bit NF4 quantization with float16 compute
and used AdamW for optimization.

## Adapter Configuration

Recorded PEFT configuration:

- **PEFT type:** LoRA
- **Task type:** causal language modeling
- **Rank (`r`):** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **Bias:** none
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **PEFT version:** 0.19.1

## Evaluation

Primary metric is whether the generated answer contains the expected value.
Exact-match percentages are also saved in `percentage_summary.csv`.

| Metric | Before unlearning | After unlearning | Change |
|---|---:|---:|---:|
| Forget accuracy, all | 95.00% | 35.00% | -60.00 pp |
| Forget held-out paraphrases | 92.50% | 34.00% | -58.50 pp |
| Forget training-identical prompts | 100.00% | 37.00% | -63.00 pp |
| Retain accuracy, all | 94.58% | 73.00% | -21.58 pp |
| Retain held-out paraphrases | 91.88% | 66.88% | -25.00 pp |
| Retain training-identical prompts | 100.00% | 85.25% | -14.75 pp |
| General controls | 56.00% | 50.00% | -6.00 pp |

Checkpoint-selection diagnostics:

| Epoch | Forget train accuracy | Retain sample accuracy | Eligible | Selection score |
|---:|---:|---:|:---:|---:|
| 1 | 82.00% | 100.00% | Yes | 18.00 |
| 2 | 62.00% | 97.00% | Yes | 35.00 |
| 3 | 37.00% | 92.00% | Yes | 55.00 |
| 4 | 25.00% | 74.00% | No | -951.00 |
| 5 | 20.00% | 76.00% | No | -944.00 |
| 6 | 15.00% | 60.00% | No | -955.00 |
| 7 | 9.00% | 68.00% | No | -941.00 |
| 8 | 7.00% | 73.00% | No | -934.00 |

Epoch 3 was selected because it gave substantial forgetting while keeping the
retain training sample above the eligibility threshold. Later epochs reduced the
forget score further but caused too much retain degradation.

## Limitations

- The result shows a clear utility trade-off: retain accuracy drops by 21.58
  percentage points.
- The method produces partial forgetting only; the post-unlearning forget score
  remains 35.00%.
- The evaluation is synthetic and prompt-based, so it does not prove robust
  removal under arbitrary paraphrases, adversarial prompting, or real-world data.
- General-control accuracy is low both before and after unlearning, so this run
  should be treated as an experimental baseline rather than a deployable model.

## Compute and Software

- **Runtime hardware recorded by notebook:** Tesla T4 GPU
- **Cloud/runtime:** Google Colab paths were used in the notebook
- **Compute region:** not recorded
- **Hours used:** not recorded
- **Carbon emitted:** not estimated
- **Transformers:** 4.48.3
- **Accelerate:** 1.3.0
- **PEFT:** 0.19.1
- **Datasets:** 3.2.0
- **Pandas:** 2.2.3
- **bitsandbytes:** 0.49.2

## Saved Artifacts

The run saves this adapter together with:

- `unlearning_config.json`
- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer files
- `results/metrics.json`
- `results/percentage_summary.csv`
- `results/unlearning_history.csv`
- before/after prediction CSV files for forget, retain, and general-control splits

## Citation

No formal citation is recorded in the saved artifacts.

## Contact

No separate model-card contact is recorded in the saved artifacts. The adapter is
part of the `unlearning-thesis` repository.

### Framework versions

- PEFT 0.19.1
