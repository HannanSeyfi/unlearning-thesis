# Week 3.5 Report: High-Accuracy Learned Baseline

## 1. Purpose

Week 3.5 created a clean, reproducible model state immediately before
unlearning. This intermediate phase separated "learning the target facts" from
"removing selected facts," which made the Week 4 comparison easier to defend.

The phase reused the fixed Week 2 synthetic data and the 50 general controls,
trained a fresh Qwen 0.5B LoRA adapter, and archived the adapter and predictions.

## 2. Configuration

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Synthetic identities: 100
- Unique facts: 500
- Forget facts: 100 across 20 identities
- Retain facts: 400 across 80 identities
- Training examples: 500
- Evaluation prompts: 1,500
- Training-identical prompts: 500
- Held-out paraphrases: 1,000
- General controls: 50
- LoRA rank: 16
- LoRA alpha: 32
- Epochs: 20
- Learning rate: `3e-4`
- Quantization: 4-bit
- Seed: 42
- Maximum sequence length: 192
- Per-device batch size: 4
- Gradient accumulation steps: 4

Only the answer portion contributed to the training loss. This prevented prompt
tokens from dominating the supervised objective.

## 3. Work Completed

### Self-contained experiment organization

The synthetic dataset and general-control files were copied under `Week 3.5` so
the baseline could be reproduced without relying on mutable notebook-generated
inputs.

### Fresh base-model evaluation

Before training, the generic model was evaluated on:

- all forget prompts;
- all retain prompts;
- held-out forget paraphrases;
- held-out retain paraphrases;
- training-identical forget prompts;
- training-identical retain prompts;
- 50 general controls.

The generic model had essentially no knowledge of the fictional facts, which
confirmed that post-training recall came from the experiment.

### LoRA training

A new adapter was trained on all 500 synthetic facts. Training runtime in the
new preserved run was approximately 1,193 seconds, and the recorded final
training loss was about `0.6103`.

### Post-training evaluation and archiving

The same evaluation prompts were run with the trained adapter. Predictions,
comparisons, metrics, tokenizer assets, and adapter weights were preserved.

## 4. Results

The newer preserved run reported:

| Evaluation group | Base | LoRA after training |
|---|---:|---:|
| Forget, all 300 prompts | 0.00% | 95.00% |
| Retain, all 1,200 prompts | 0.08% | 94.67% |
| Forget held-out paraphrases | 0.00% | 92.50% |
| Retain held-out paraphrases | 0.00% | 92.00% |
| Forget training-identical prompts | 0.00% | 100.00% |
| Retain training-identical prompts | 0.25% | 100.00% |

The archived earlier reference run reported 94.33% forget and 93.92% retain
overall. The small difference between preserved runs is consistent with
quantized inference and rerun variation.

### General controls

The newer Week 3.5 metrics originally reported:

- base general-control contains-value score: 84%;
- LoRA general-control contains-value score: 62%.

Later boundary-aware rescoring in Week 4 found the comparable learned-model
general score to be 56%. The earlier substring scorer incorrectly accepted
cases such as an expected `56` appearing inside `560`.

This correction matters: Week 3.5 successfully learned the synthetic facts, but
general behavior was already degraded before unlearning began.

## 5. Interpretation

Week 3.5 achieved its main purpose. The model strongly learned both target
groups and generalized beyond exact training wording:

- held-out forget accuracy reached 92.5%;
- held-out retain accuracy reached 92.0%;
- seen-prompt accuracy reached 100%.

These results provided enough headroom for a meaningful unlearning experiment.
If forget accuracy had been low before unlearning, a later reduction would not
have demonstrated removal of learned information.

The general-control decline also established an important baseline condition:
Week 4 should not attribute all low general performance to unlearning, because
much of the damage occurred during synthetic-fact learning.

## 6. Decisions Made

- Use this adapter as the source model for Week 4.
- Keep seen and held-out prompt results separate.
- Archive the complete adapter rather than only summary metrics.
- Retain row-level predictions for direct pre/post comparison.
- Use boundary-aware answer matching in subsequent analysis.

## 7. Deliverables

- `Week 3.5/notebooks/week3_5_train_high_accuracy_baseline.ipynb`
- Self-contained synthetic and general-control data
- Archived successful adapter and tokenizer files
- Base and LoRA prediction CSVs
- Row-level comparison CSV
- Metrics and percentage summaries

## 8. Limitations

- General capability remained weak with the adapter enabled.
- The original contains-value scorer inflated some general results.
- Quantized reruns can vary slightly.
- High output accuracy demonstrates accessible learned behavior, not a precise
  map of where facts are stored in model parameters.

## 9. Transition to Week 4

Week 4 loaded the Week 3.5 adapter and updated only its LoRA parameters. The
unlearning objective increased loss on forget examples while simultaneously
decreasing loss on sampled retain examples.

## 10. Evidence Used

- `Week 3.5/README.md`
- `Week 3.5/notebooks/week3_5_train_high_accuracy_baseline.ipynb`
- `Week 3.5/results/reference_successful_run`
- `imports/2026-06-14-week3.5/qwen05_high_accuracy_baseline/metrics.json`
- Imported prediction and comparison CSV files
