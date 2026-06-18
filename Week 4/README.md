# Week 4: Gradient-Ascent Unlearning

This is the first unlearning stage. It starts from the learned Qwen 0.5B LoRA
adapter produced by the strict Week 3.5 baseline.

The optimization objective performs gradient ascent on the designated forget
examples and gradient descent on sampled retain examples:

`objective = -forget_loss + retain_weight * retain_loss`

Only LoRA adapter parameters are updated. The final experiment reports:

- Forget accuracy before and after unlearning
- Retain accuracy before and after unlearning
- General-control accuracy before and after unlearning
- Detailed predictions, checkpoint diagnostics, metrics, and the unlearned
  adapter

The synthetic evaluation set has 1,500 rows: 500 training-identical prompts
and 1,000 held-out paraphrases. Results preserve a prompt-overlap flag so these
can be reported separately.

Run:

`notebooks/week4_gradient_ascent_unlearning.ipynb`

Before running Week 4, run the strict Week 3.5 baseline:

`Week 3.5/notebooks/week3_5_train_high_accuracy_baseline_strict.ipynb`

Week 4 requires:

`MyDrive/Thesis/Week 3.5/results/qwen05_high_accuracy_baseline/adapter`

and:

`MyDrive/Thesis/Week 3.5/results/qwen05_high_accuracy_baseline/metrics.json`

The metrics file must contain `strict_scoring` metadata. The notebook no longer
falls back to the older archived reference adapter.

The previous joint-training work is preserved separately under:

`Week 4 - Joint Training Experiments`
