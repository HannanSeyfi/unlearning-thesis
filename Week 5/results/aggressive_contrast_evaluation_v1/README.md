# Week 5 Aggressive Contrast Evaluation Results

This directory contains the aggressive Week 5 contrast evaluation outputs.

The Colab notebook in:

`Week 5/aggressive_contrast_evaluation/week5_aggressive_contrast_eval_colab.ipynb`

writes each checkpoint evaluation into its own subfolder here. The completed
default run is:

`c09_most_aggressive_epoch_03/results`

See the generated report:

`c09_most_aggressive_epoch_03/results/aggressive_contrast_report.md`

This run shows that the aggressive checkpoint reaches Week 4-level forgetting
but does so by damaging retain and general-control performance. It supports the
Week 6 target: recover stronger forgetting while avoiding global retain
collapse.

This keeps the selected Week 5 preserving run intact under:

`Week 5/results/retain_regularized_unlearning_resumable_v1`

