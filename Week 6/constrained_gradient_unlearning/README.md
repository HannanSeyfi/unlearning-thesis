# Constrained Gradient Unlearning Runner

This folder contains the Week 6 runner:

`train_week6_constrained_gradient_unlearning.py`

The runner performs a focused sweep of PCGrad-style constrained unlearning
checkpoints, selects the best retain-eligible checkpoint, evaluates it on the
full forget, retain, and general-control sets, and writes thesis-ready CSV,
JSON, adapter, and Markdown outputs under:

`Week 6/results/constrained_gradient_unlearning_v1`

The Colab notebook in `../notebooks` is the recommended entrypoint because it
sets up GitHub-backed persistence and can push progress after each epoch.
