---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- synthetic-facts
- machine-unlearning
- thesis
---

# Week 3.5 High-Accuracy Qwen 0.5B Synthetic-Fact LoRA Adapter

This is the strict-scored Week 3.5 PEFT/LoRA adapter used as the learned baseline for later machine-unlearning experiments in this thesis repository. It adapts `Qwen/Qwen2.5-0.5B-Instruct` to answer a controlled synthetic-fact dataset about fictional identities, then serves as the starting checkpoint for Week 4 and Week 5 unlearning runs.

## Model Details

### Model Description

- **Developed by:** Hannan Seyfi
- **Funded by:** Academic thesis work; no external funder is recorded in this repository.
- **Shared by:** HannanSeyfi via the `HannanSeyfi/unlearning-thesis` GitHub repository
- **Model type:** PEFT LoRA adapter for causal language modeling / text generation
- **Language(s) (NLP):** English synthetic question answering; the base model is multilingual
- **License:** No separate adapter license file is declared in this repository. The base model card declares `apache-2.0` for `Qwen/Qwen2.5-0.5B-Instruct`.
- **Finetuned from model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Run name:** `week3_5_qwen05_high_accuracy_baseline_strict`
- **Created at:** `2026-06-22T09:27:13.602334+00:00`

### Model Sources

- **Repository:** `https://github.com/HannanSeyfi/unlearning-thesis`
- **Adapter path:** `Week 3.5/results/qwen05_high_accuracy_baseline/adapter`
- **Training notebook:** `Week 3.5/notebooks/week3_5_train_high_accuracy_baseline_strict.ipynb`
- **Strict re-evaluation notebook:** `Week 3.5/notebooks/week3_5_strict_reevaluate_learned_model.ipynb`
- **Base model:** `https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct`
- **Paper:** No separate paper is published for this adapter.
- **Demo:** No hosted demo is provided.

## Uses

### Direct Use

Use this adapter to reproduce the Week 3.5 learned synthetic-fact baseline. It is intended for controlled research evaluation, especially comparisons between the learned baseline and later unlearned adapters.

### Downstream Use

Week 4 and Week 5 experiments load this adapter as the starting point for gradient-ascent and retain-regularized unlearning. The adapter can also be used to regenerate strict baseline metrics against the repository's synthetic and general-control evaluation files.

### Out-of-Scope Use

This adapter is not intended as a general-purpose assistant, production model, or factual QA model. Its training target is a small synthetic dataset of fictional identities, and its behavior outside that controlled setup is not evaluated here.

## Bias, Risks, and Limitations

The adapter is intentionally specialized to synthetic facts. It can overfit to the synthetic training distribution and degrade general knowledge behavior. In the recorded strict run, general-control contains-value accuracy dropped from 88.0% for the base model to 56.0% after LoRA training.

### Recommendations

Use the adapter only with the matching Week 3.5 evaluation protocol. Compare both synthetic-fact accuracy and general-control preservation before drawing conclusions about unlearning quality.

## How to Get Started with the Model

From a clone of this repository:

```python
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_dir = Path("Week 3.5/results/qwen05_high_accuracy_baseline/adapter")

tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_dir)
model.eval()
```

## Training Details

### Training Data

- **Dataset:** `Week 3.5/data/synthetic_facts_v1`
- **Fictional identities:** 100
- **Unique synthetic facts:** 500
- **Forget facts:** 100 facts from 20 identities
- **Retain facts:** 400 facts from 80 identities
- **Training examples:** 500
- **Training target:** fact value only
- **Evaluation leakage prevention:** enabled

The fixed general-control questions are stored in `Week 3.5/data/general_controls_v1` and are held out from training and selection.

### Training Procedure

The adapter was trained in the strict Week 3.5 baseline notebook and evaluated after save/reload using the Week 4-compatible scorer.

#### Training Hyperparameters

- **Seed:** 42
- **Training regime:** 4-bit base-model loading with LoRA adapter training
- **Epochs:** 20
- **Learning rate:** `3e-4`
- **Per-device batch size:** 4
- **Gradient accumulation steps:** 4
- **Max sequence length:** 192
- **LoRA rank:** 16
- **LoRA alpha:** 32
- **LoRA dropout:** 0.05
- **LoRA bias:** `none`
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

#### Speeds, Sizes, Times

- **Train runtime:** 1278.8835 seconds
- **Train samples/second:** 7.819
- **Train steps/second:** 0.485
- **Train loss:** 0.6102708791675526
- **Reported epoch:** 19.384
- **Total FLOPs:** 932461804631040.0

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- **Synthetic forget evaluation examples:** 300
- **Synthetic retain evaluation examples:** 1,200
- **Training-identical evaluation prompts:** 500
- **Held-out paraphrase prompts:** 1,000
- **General-control questions:** 50

#### Factors

Evaluation is separated by model stage (`base_before_training`, `lora_after_training`), synthetic split (`forget`, `retain`), seen training prompts versus held-out paraphrases, and general-control questions.

#### Metrics

- **Exact match:** normalized generated answer equals the expected answer.
- **Contains value:** case-insensitive normalized whole-token boundary match against the expected value.
- **Generation:** deterministic evaluation with `max_new_tokens=16`.

### Results

| Model stage | Test set | Split | Questions | Contains-value correct | Contains-value % | Exact-match % |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Base before training | General control | general | 50 | 44 | 88.0 | 6.0 |
| Base before training | Synthetic facts | forget | 300 | 0 | 0.0 | 0.0 |
| Base before training | Synthetic facts | retain | 1,200 | 1 | 0.0833 | 0.0 |
| LoRA after training | General control | general | 50 | 28 | 56.0 | 50.0 |
| LoRA after training | Synthetic facts | forget | 300 | 285 | 95.0 | 95.0 |
| LoRA after training | Synthetic facts | retain | 1,200 | 1,135 | 94.5833 | 94.5833 |

#### Held-Out and Seen Prompt Results

| Model stage | Subset | Questions | Exact match | Contains value |
| --- | --- | ---: | ---: | ---: |
| Base before training | Forget held-out paraphrases | 200 | 0.0 | 0.0 |
| Base before training | Retain held-out paraphrases | 800 | 0.0 | 0.0 |
| Base before training | Forget seen prompts | 100 | 0.0 | 0.0 |
| Base before training | Retain seen prompts | 400 | 0.0 | 0.0025 |
| LoRA after training | Forget held-out paraphrases | 200 | 0.925 | 0.925 |
| LoRA after training | Retain held-out paraphrases | 800 | 0.91875 | 0.91875 |
| LoRA after training | Forget seen prompts | 100 | 1.0 | 1.0 |
| LoRA after training | Retain seen prompts | 400 | 1.0 | 1.0 |

#### Summary

The LoRA adapter successfully learns the synthetic facts under strict scoring, reaching 95.0% contains-value accuracy on the forget split and 94.5833% on the retain split. The same run also shows a general-control preservation cost: contains-value accuracy on the fixed general-control questions is 56.0% after training.

## Environmental Impact

- **Hardware Type:** Google Colab GPU runtime; exact accelerator type is not recorded in `metrics.json`.
- **Hours used:** approximately 0.355 hours for the recorded training runtime.
- **Cloud Provider:** Google Colab
- **Compute Region:** Not recorded
- **Carbon Emitted:** Not estimated for this run

## Technical Specifications

### Model Architecture and Objective

The base model is `Qwen/Qwen2.5-0.5B-Instruct`, a causal language model. This artifact is a PEFT LoRA adapter with task type `CAUSAL_LM`, trained to produce synthetic fact values from prompts about fictional identities.

### Compute Infrastructure

The training notebook was run in Colab from a cloned copy of this GitHub repository at `/content/unlearning-thesis`.

#### Hardware

A GPU runtime was used for adapter training. The exact GPU model was not recorded in the saved metrics.

#### Software

- **PEFT:** 0.19.1
- **Transformers:** used by the training notebooks
- **Adapter format:** PEFT LoRA safetensors

## Citation

If citing the base model, use the citation provided by the Qwen model card. If citing this adapter, cite the repository and the Week 3.5 run path:

```text
Seyfi, H. (2026). Unlearning Thesis: Week 3.5 high-accuracy Qwen 0.5B synthetic-fact LoRA baseline. https://github.com/HannanSeyfi/unlearning-thesis
```

## Glossary

- **Forget split:** synthetic facts later targeted by unlearning experiments.
- **Retain split:** synthetic facts intended to remain available after unlearning.
- **General control:** fixed held-out general-knowledge questions used to check behavior preservation.
- **Strict scoring:** deterministic generation plus normalized whole-token matching to avoid substring false positives.

## More Information

See `Week 3.5/README.md` and `Week 3.5/results/qwen05_high_accuracy_baseline/metrics.json` for the full run context.

## Model Card Authors

Hannan Seyfi, with metadata filled from the saved Week 3.5 run artifacts.

## Model Card Contact

Open an issue in `HannanSeyfi/unlearning-thesis` for repository-level questions.

### Framework versions

- PEFT 0.19.1
