# Week 3.5 Report: High-Accuracy Learned Baseline

Generated from saved result files.

## Purpose

Week 3.5 created the learned model state used as the starting point for unlearning. The goal was to train a fresh LoRA adapter on the synthetic facts and confirm that the model had actually learned both forget and retain facts before any unlearning was attempted.

## Configuration

- Run name: `week3_5_qwen05_high_accuracy_baseline`
- Created at UTC: `2026-06-13T18:59:44.537674+00:00`
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Training examples: 500
- Forget evaluation examples: 300
- Retain evaluation examples: 1200
- Held-out paraphrase prompts: 1000
- General-control questions: 50
- Epochs: 20
- Learning rate: 0.0003
- 4-bit loading: True
- Train on answer value only: True
- Evaluation leakage prevented: True

## Results

| Evaluation group | Examples | Base before training | LoRA after training |
|---|---|---|---|
| Forget, all prompts | 300 | 0.00% | 95.00% |
| Retain, all prompts | 1200 | 0.08% | 94.67% |
| Forget held-out paraphrases | 200 | 0.00% | 92.50% |
| Retain held-out paraphrases | 800 | 0.00% | 92.00% |
| Forget seen prompts | 100 | 0.00% | 100.00% |
| Retain seen prompts | 400 | 0.25% | 100.00% |
| General controls | 50 | 84.00% | 62.00% |

## Interpretation

The Week 3.5 adapter learned the synthetic facts strongly. Overall forget accuracy reached 95.00%, and overall retain accuracy reached 94.67%. Performance also transferred to held-out paraphrases, which indicates the model learned more than one exact prompt wording.

The main caveat is general behavior. The saved Week 3.5 scorer reports general-control performance falling from 84.00% to 62.00%. Later Week 4 analysis used stricter boundary-aware matching and treated the comparable pre-unlearning general score as 56.00%.

## Files Used

- `imports\2026-06-14-week3.5\qwen05_high_accuracy_baseline\metrics.json`
- `Week 3.5\results\reference_successful_run\results\percentage_summary.csv`
