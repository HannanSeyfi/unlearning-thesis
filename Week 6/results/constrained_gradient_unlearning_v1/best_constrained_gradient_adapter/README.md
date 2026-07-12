---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- constrained-gradient-unlearning
- pcgrad
---

# Week 6 Constrained Gradient Unlearning Adapter

This is the selected PEFT LoRA adapter from the Week 6 constrained-gradient
unlearning experiment in the `unlearning-thesis` project. It starts from the
strict-scored Week 3.5 high-accuracy LoRA adapter for
`Qwen/Qwen2.5-0.5B-Instruct` and uses a PCGrad-style constrained update rule.

The selected Week 6 adapter improves forgetting slightly relative to the Week 5
preservation-oriented checkpoint, while keeping retain accuracy much higher than
the aggressive forgetting baselines. It does not reach the target forgetting
band: forget held-out accuracy remains 64.00%.

## Model Details

- **Run name:** `constrained_gradient_unlearning_v1`
- **Created at UTC:** `2026-06-23T19:33:26.892768+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT LoRA adapter for causal language modeling
- **Source adapter:** Week 3.5 strict high-accuracy baseline adapter
- **Selected candidate:** `c02_pcgrad_balanced`
- **Selected epoch:** 4
- **Method:** PCGrad-style constrained forget ascent with retain cross-entropy and retain KL preservation
- **Developed for:** synthetic-fact machine-unlearning research
- **Language:** English prompts in the saved synthetic and control evaluations
- **License:** not specified in the saved experiment artifacts

## Intended Use

This adapter is intended for research on selective synthetic-fact unlearning and
gradient-conflict control. It is useful as the Week 6 constrained-update
checkpoint: it tests whether forget-ascent gradients can be projected away from
retain-preservation gradients when those updates conflict.

It is not intended for production use, safety-critical decisions, or claims of
complete knowledge deletion. The benchmark uses fictional synthetic people and
controlled prompts, so these results should not be treated as evidence about
real personal-data removal.

## How to Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "Week 6/results/constrained_gradient_unlearning_v1/best_constrained_gradient_adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_dir)
```

## Training and Selection Data

The run uses the Week 3.5 synthetic-fact and general-control data:

- **Forget train:** 100 examples
- **Retain train:** 400 examples
- **Forget evaluation:** 300 prompts
- **Retain evaluation:** 1,200 prompts
- **General controls:** 50 prompts
- **Checkpoint-selection forget held-out sample:** 80 prompts
- **Checkpoint-selection retain held-out sample:** 160 prompts

The saved result files preserve whether a prompt was seen in original training
and whether it was used for checkpoint selection.

## Training Procedure

Only LoRA adapter parameters are updated. The base model remains frozen. For
each update, the runner computes:

- a forget-ascent gradient from `-forget_loss`
- a preservation gradient from `retain_weight * retain_loss + kl_weight * retain_kl`

When these gradients conflict, the forget gradient is projected away from the
preservation gradient before the optimizer step. This is a PCGrad-style
constraint: destructive forget updates are damped, while compatible forget
updates remain available.

Recorded run settings:

- **Focused sweep candidates:** 6
- **Full grid enabled:** false
- **Resume enabled:** true
- **Maximum epochs per candidate:** 6
- **Batch size:** 1
- **Gradient accumulation steps:** 8
- **Maximum gradient norm:** `1.0`
- **Target forget maximum:** `20.0`
- **Retain selection threshold:** `85.0%`
- **Target forget held-out threshold:** `45.0%`

Selected candidate configuration:

- **Candidate:** `c02_pcgrad_balanced`
- **Epoch:** 4
- **Learning rate:** `2e-5`
- **Retain weight:** `2.0`
- **KL weight:** `0.5`
- **Forget scale:** `1.0`
- **Projection strength:** `1.0`
- **Mean forget loss:** `0.446781540501579`
- **Mean retain loss:** `0.0105380254459851`
- **Mean KL loss:** `0.0102115526158286`
- **Mean gradient cosine:** `-0.0038261883192793`
- **Conflict rate:** `0.56`
- **Projection rate:** `0.56`
- **Forget held-out selection accuracy:** `61.25%`
- **Retain held-out selection accuracy:** `87.50%`
- **Selection score:** `39.625`

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

| Metric | Before unlearning | After Week 6 | Change |
|---|---:|---:|---:|
| Forget accuracy, all | 95.00% | 67.33% | -27.67 pp |
| Forget held-out paraphrases | 92.50% | 64.00% | -28.50 pp |
| Forget all, excluding selection prompts | not applicable | 69.55% | not applicable |
| Retain accuracy, all | 94.58% | 90.92% | -3.67 pp |
| Retain held-out paraphrases | 91.88% | 87.00% | -4.88 pp |
| Retain all, excluding selection prompts | not applicable | 91.44% | not applicable |
| General controls | 56.00% | 54.00% | -2.00 pp |

Detailed split summary:

| Stage | Split | Seen in original training | Used for selection | Correct / total | Accuracy |
|---|---|:---:|:---:|---:|---:|
| Before | Forget | No | No | 112 / 120 | 93.33% |
| Before | Forget | No | Yes | 73 / 80 | 91.25% |
| Before | Forget | Yes | No | 100 / 100 | 100.00% |
| After | Forget | No | No | 79 / 120 | 65.83% |
| After | Forget | No | Yes | 49 / 80 | 61.25% |
| After | Forget | Yes | No | 74 / 100 | 74.00% |
| Before | Retain | No | No | 588 / 640 | 91.88% |
| Before | Retain | No | Yes | 147 / 160 | 91.88% |
| Before | Retain | Yes | No | 400 / 400 | 100.00% |
| After | Retain | No | No | 556 / 640 | 86.88% |
| After | Retain | No | Yes | 140 / 160 | 87.50% |
| After | Retain | Yes | No | 395 / 400 | 98.75% |
| Before | General | No | No | 28 / 50 | 56.00% |
| After | General | No | No | 27 / 50 | 54.00% |

## Candidate Ranking

Best checkpoint per focused-sweep candidate:

| Candidate | Epoch | LR | Retain weight | KL weight | Forget scale | Projection | Forget selection | Retain selection | Conflict | Projection rate | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c02_pcgrad_balanced` | 4 | `2e-5` | 2.0 | 0.5 | 1.0 | 1.0 | 61.25% | 87.50% | 0.56 | 0.56 | 39.62 |
| `c06_no_projection_control` | 4 | `2e-5` | 2.0 | 1.0 | 1.0 | 0.0 | 67.50% | 86.88% | 0.64 | 0.00 | 33.16 |
| `c01_pcgrad_higher_kl` | 4 | `2e-5` | 2.0 | 1.0 | 1.0 | 1.0 | 70.00% | 86.25% | 0.69 | 0.69 | 30.44 |
| `c03_pcgrad_aggressive_guarded` | 2 | `5e-5` | 2.0 | 1.0 | 0.75 | 1.0 | 72.50% | 86.88% | 0.63 | 0.63 | 28.16 |
| `c05_pcgrad_retain_heavy` | 4 | `2e-5` | 4.0 | 0.5 | 1.5 | 1.0 | 73.75% | 85.00% | 0.60 | 0.60 | 26.25 |
| `c04_pcgrad_preserve_high` | 6 | `1e-5` | 4.0 | 1.0 | 1.25 | 1.0 | 77.50% | 87.50% | 0.57 | 0.57 | 23.38 |

## Comparison to Earlier Runs

| Model stage | Forget all | Forget held-out | Retain all | Retain held-out | General |
|---|---:|---:|---:|---:|---:|
| Before unlearning, Week 3.5 adapter | 95.00% | 92.50% | 94.58% | 91.88% | 56.00% |
| Week 4 gradient ascent | 35.00% | 34.00% | 73.00% | 66.88% | 50.00% |
| Week 5 preserving selected | 69.67% | 67.00% | 92.42% | 88.88% | 58.00% |
| Week 5 aggressive c09 epoch 03 | 32.67% | 32.00% | 67.75% | 61.88% | 46.00% |
| Week 6 constrained gradient selected | 67.33% | 64.00% | 90.92% | 87.00% | 54.00% |

Lower is better for forget metrics; higher is better for retain and general
metrics. Week 6 improves forgetting slightly relative to the Week 5 preserving
checkpoint, but it does not approach Week 4 or the Week 5 aggressive checkpoint.
It preserves retain behavior far better than those aggressive baselines.

## Limitations

- Forgetting is partial. After Week 6, forget accuracy remains 67.33% overall
  and 64.00% on held-out paraphrases.
- The selected checkpoint misses the Week 6 target band of 45.00% forget
  held-out accuracy or lower.
- Retain preservation is better than aggressive forgetting baselines but lower
  than the Week 5 preserving selected checkpoint.
- General-control accuracy falls from 56.00% before unlearning to 54.00% after
  Week 6.
- The evaluation is synthetic and prompt-based, so it does not prove robust
  removal under arbitrary paraphrases, adversarial prompting, or real-world data.

## Compute and Software

- **Runtime entry point:** GitHub-backed Colab notebook and local Python runner
- **Runtime hardware:** not recorded in the saved artifacts
- **Compute region:** not recorded
- **Hours used:** not recorded
- **Carbon emitted:** not estimated
- **PEFT:** 0.19.1
- **Other package versions:** not recorded in the saved artifacts

The Colab notebook installs `transformers`, `peft`, `accelerate`,
`bitsandbytes`, `pandas`, `numpy`, and `safetensors`, but the saved artifacts do
not record exact installed versions except for PEFT in `adapter_config.json`.

## Saved Artifacts

The run saves this adapter together with:

- `adapter_config.json`
- `adapter_model.safetensors`
- tokenizer files
- `results/metrics.json`
- `results/percentage_summary.csv`
- `results/sweep_history.csv`
- `results/candidate_best_summary.csv`
- `results/week4_week5_week6_comparison.csv`
- `results/week6_constrained_gradient_report.md`
- before/after prediction CSV files for forget, retain, and general-control splits
- final evaluation files excluding checkpoint-selection examples
- resumable state under `resume_state/`

## Citation

No formal citation is recorded in the saved artifacts.

## Contact

No separate model-card contact is recorded in the saved artifacts. The adapter is
part of the `unlearning-thesis` repository.

### Framework versions

- PEFT 0.19.1
