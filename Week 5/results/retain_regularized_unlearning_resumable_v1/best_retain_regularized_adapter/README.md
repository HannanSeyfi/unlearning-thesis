---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- machine-unlearning
- retain-regularized-unlearning
---

# Week 5 Retain-Regularized Unlearning Adapter

This is the selected PEFT LoRA adapter from the Week 5 retain-regularized
unlearning experiment in the `unlearning-thesis` project. It starts from the
strict-scored Week 3.5 high-accuracy LoRA adapter for
`Qwen/Qwen2.5-0.5B-Instruct` and applies a preservation-oriented unlearning
objective.

Week 5 did not forget as strongly as the Week 4 gradient-ascent baseline, but it
preserved retain and general-control behavior much better. Forget held-out
accuracy dropped from 92.50% to 67.00%, while retain held-out accuracy stayed at
88.88%.

## Model Details

- **Run name:** `week5_retain_regularized_unlearning_resumable_v1`
- **Created at UTC:** `2026-06-22T20:23:51.868432+00:00`
- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Adapter type:** PEFT LoRA adapter for causal language modeling
- **Source adapter:** Week 3.5 strict high-accuracy baseline adapter
- **Source run:** `week3_5_qwen05_high_accuracy_baseline_strict`
- **Selected candidate:** `c07_higher_kl`
- **Selected epoch:** 3
- **Method:** gradient ascent on forget loss plus retain cross-entropy and retain KL regularization against the original Week 3.5 adapter
- **Developed for:** synthetic-fact machine-unlearning research
- **Language:** English prompts in the saved synthetic and control evaluations
- **License:** not specified in the saved experiment artifacts

## Intended Use

This adapter is intended for research and analysis of the forgetting-versus-
retention trade-off in synthetic-fact unlearning. It is useful as the Week 5
preservation-oriented checkpoint: retain/general behavior is largely preserved,
but many forget facts remain recoverable.

It is not intended for production use, safety-critical decisions, or claims of
complete knowledge deletion. The benchmark uses fictional synthetic people and
controlled prompts, so these results should not be treated as evidence about
real personal-data removal.

## How to Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = "Week 5/results/retain_regularized_unlearning_resumable_v1/best_retain_regularized_adapter"

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

Only LoRA adapter parameters are updated. The base model remains frozen. The
objective used by the Week 5 notebook is:

```text
objective = -forget_loss + retain_weight * retain_loss + kl_weight * retain_kl
```

The KL term compares the trainable Week 5 adapter against the frozen original
Week 3.5 adapter on retain answer tokens.

Recorded run settings:

- **Focused sweep candidates:** 9
- **Full grid enabled:** false
- **Resume enabled:** true
- **Maximum epochs per candidate:** 8
- **Batch size:** 1
- **Gradient accumulation steps:** 8
- **Maximum gradient norm:** `1.0`
- **Target forget maximum:** `20.0`
- **Retain selection threshold:** `85.0%`

Selected candidate configuration:

- **Candidate:** `c07_higher_kl`
- **Epoch:** 3
- **Learning rate:** `2e-5`
- **Retain weight:** `2.0`
- **KL weight:** `1.0`
- **Forget held-out selection accuracy:** `63.75%`
- **Retain held-out selection accuracy:** `88.125%`
- **Selection score:** `37.03125`

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

The strict scorer uses a case-insensitive normalized whole-token regex boundary
match. Normalization applies NFKD, removes combining marks, lowercases,
normalizes whitespace, and strips outer punctuation. Generation used
`max_new_tokens=16`.

Primary metric is whether the generated answer contains the expected value.
Exact-match percentages are also saved in `percentage_summary.csv`.

| Metric | Before unlearning | After Week 5 | Change |
|---|---:|---:|---:|
| Forget accuracy, all | 95.00% | 69.67% | -25.33 pp |
| Forget held-out paraphrases | 92.50% | 67.00% | -25.50 pp |
| Forget all, excluding selection prompts | not applicable | 71.36% | not applicable |
| Retain accuracy, all | 94.58% | 92.42% | -2.17 pp |
| Retain held-out paraphrases | 91.88% | 88.88% | -3.00 pp |
| Retain all, excluding selection prompts | not applicable | 93.08% | not applicable |
| General controls | 56.00% | 58.00% | +2.00 pp |

Detailed split summary:

| Stage | Split | Seen in original training | Used for selection | Correct / total | Accuracy |
|---|---|:---:|:---:|---:|---:|
| Before | Forget | No | No | 112 / 120 | 93.33% |
| Before | Forget | No | Yes | 73 / 80 | 91.25% |
| Before | Forget | Yes | No | 100 / 100 | 100.00% |
| After | Forget | No | No | 82 / 120 | 68.33% |
| After | Forget | No | Yes | 52 / 80 | 65.00% |
| After | Forget | Yes | No | 75 / 100 | 75.00% |
| Before | Retain | No | No | 588 / 640 | 91.88% |
| Before | Retain | No | Yes | 147 / 160 | 91.88% |
| Before | Retain | Yes | No | 400 / 400 | 100.00% |
| After | Retain | No | No | 570 / 640 | 89.06% |
| After | Retain | No | Yes | 141 / 160 | 88.12% |
| After | Retain | Yes | No | 398 / 400 | 99.50% |
| Before | General | No | No | 28 / 50 | 56.00% |
| After | General | No | No | 29 / 50 | 58.00% |

## Candidate Ranking

Best checkpoint per focused-sweep candidate:

| Candidate | Epoch | LR | Retain weight | KL weight | Forget selection | Retain selection | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c07_higher_kl` | 3 | `2e-5` | 2.0 | 1.0 | 63.75% | 88.12% | 37.03 |
| `c01_low_lr_balanced` | 8 | `1e-5` | 2.0 | 0.5 | 65.00% | 85.00% | 35.00 |
| `c08_most_preserving` | 8 | `1e-5` | 4.0 | 1.0 | 66.25% | 86.88% | 34.22 |
| `c04_lower_retain` | 3 | `2e-5` | 1.0 | 0.5 | 67.50% | 87.50% | 33.12 |
| `c06_lower_kl` | 5 | `2e-5` | 2.0 | 0.1 | 67.50% | 86.88% | 32.97 |
| `c02_mid_lr_balanced` | 3 | `2e-5` | 2.0 | 0.5 | 70.00% | 86.88% | 30.47 |
| `c05_higher_retain` | 4 | `2e-5` | 4.0 | 0.5 | 71.25% | 87.50% | 29.38 |
| `c09_most_aggressive` | 1 | `5e-5` | 1.0 | 0.1 | 83.75% | 93.12% | 18.28 |
| `c03_high_lr_balanced` | 1 | `5e-5` | 2.0 | 0.5 | 83.75% | 88.75% | 17.19 |

## Comparison to Week 4

Week 5 is a preservation-oriented checkpoint. Compared with Week 4, it forgets
less but preserves much more retain/general behavior:

| Metric | Week 4 after | Week 5 after | Week 5 minus Week 4 |
|---|---:|---:|---:|
| Forget all | 35.00% | 69.67% | +34.67 pp |
| Forget held-out paraphrases | 34.00% | 67.00% | +33.00 pp |
| Retain all | 73.00% | 92.42% | +19.42 pp |
| Retain held-out paraphrases | 66.88% | 88.88% | +22.00 pp |
| General controls | 50.00% | 58.00% | +8.00 pp |

Lower is better for forget metrics; higher is better for retain and general
metrics. Week 4 is the stronger forgetting baseline, while Week 5 is the
stronger retention baseline.

## Limitations

- Forgetting is partial. After Week 5, forget accuracy remains 69.67% overall
  and 67.00% on held-out paraphrases.
- The selected checkpoint intentionally prioritizes retain preservation, so it
  should not be described as the strongest forgetting result.
- The evaluation is synthetic and prompt-based, so it does not prove robust
  removal under arbitrary paraphrases, adversarial prompting, or real-world data.
- The run maps a trade-off curve rather than solving isolated unlearning.

## Compute and Software

- **Runtime hardware recorded by notebook:** Tesla T4 GPU
- **Runtime environment:** Google Colab paths were used in the notebook
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
- `results/sweep_history.csv`
- `results/candidate_best_summary.csv`
- `results/week4_week5_comparison.csv`
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
