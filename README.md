# Unlearning Thesis

Week-by-week Colab notebooks, synthetic data, reports, and selected result
artifacts for a machine-unlearning thesis project.

This repository is arranged so notebooks can be opened directly in Google
Colab from GitHub. The notebooks use GitHub, not Google Drive, for persistent
data and outputs: each notebook clones the repo into `/content/unlearning-thesis`
and commits its output folder back to GitHub.

## Quick Start

1. Open the one-time setup notebook in Colab:
   [setup_colab_from_github.ipynb](https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/main/setup_colab_from_github.ipynb)
2. In Colab Secrets, add `GITHUB_TOKEN` with **Contents: Read and write**
   access for this repository.
3. Run all cells in the setup notebook. It clones or updates this repository at
   `/content/unlearning-thesis`.
4. Open notebooks from [COLAB_NOTEBOOKS.md](COLAB_NOTEBOOKS.md).

If you publish the repo under a different name, replace
`HannanSeyfi/unlearning-thesis` in this README, `COLAB_NOTEBOOKS.md`, and
`setup_colab_from_github.ipynb`.

## Project Structure

- `Week 1/notebooks`: initial Colab language-model pipeline.
- `Week 2/notebooks`: synthetic-fact dataset generation.
- `Week 2/data/synthetic_facts_v1`: generated synthetic training and
  evaluation data.
- `Week 3/notebooks`: LoRA baselines, balanced controls, and routed dual-target
  experiments.
- `Week 3/data/general_controls_v1`: versioned general control, replay, and
  validation sets.
- `Week 3/data/always_on_preservation_v2`: expanded replay and validation sets
  for always-on LoRA selection.
- `Week 3/results`: preserved result summaries and analysis extracts.
- `Week 3.5`: strict-scored high-accuracy Qwen 0.5B LoRA baseline, fixed
  general controls, source data, and archived successful adapter.
- `Week 4`: gradient-ascent unlearning of the strict Week 3.5 Qwen 0.5B
  adapter.
- `Week 5`: retain-regularized unlearning sweep with KL preservation against
  the Week 3.5 adapter.
- `Week 6`: constrained-gradient unlearning with PCGrad-style forget-gradient
  projection against retain-preservation gradients.
- `Week 7`: adaptive constrained unlearning with dynamic forget pressure,
  explicit retain/general floors, a matched fixed-pressure control, and a
  separately versioned rollback-constrained follow-up.
- `Week 4 - Joint Training Experiments`: preserved 700-example joint-training
  experiments that previously occupied the Week 4 name.
- `reports`: generated thesis progress reports and report-generation notebooks.
- `Tools/local_colab`: optional archived local-runtime scripts.

## Running Order

For a fresh Colab runtime, use this order:

1. `setup_colab_from_github.ipynb`
2. Week 1 and Week 2 notebooks if you want to regenerate the initial pipeline
   and synthetic data.
3. Week 3 notebooks for LoRA baselines and controls.
4. Week 3.5 strict baseline or strict re-evaluation.
5. Week 4 gradient-ascent unlearning.
6. Week 5 retain-regularized unlearning.
7. Week 6 constrained-gradient unlearning.
8. Week 7 adaptive constrained unlearning.
9. Week 7 rollback-constrained v2 follow-up.
10. Report notebooks in `reports`.

Each notebook pulls the latest GitHub state before writing and pushes its output
folder at the end. The Week 5 resumable, Week 6, and Week 7 notebooks can also
push resume checkpoints during their sweeps.

## Colab Notes

Most training notebooks install their own dependencies in the first code cell.
Use a GPU runtime for the model-training notebooks. Data-generation and report
notebooks can usually run on CPU.

The fixed Week 3 general controls are stored as files rather than generated
inside a notebook. `manifest.json` records the SHA-256 hash of each JSONL file.
After the setup notebook clones the repo, the Week 3 notebooks can reuse those
files without manual upload.

## GitHub Notes

This GitHub-ready copy intentionally excludes duplicate upload mirrors, import
mirrors, zip snapshots, and hidden analysis copies from the local workspace.
The repository keeps the canonical week folders, reports, analysis notes, and
selected artifacts needed for the current Colab workflow.

GitHub rejects individual files larger than about 100 MB. The Colab sync helper
uses a 95 MB safety limit and will stop with a clear error if a generated file is
too large for normal GitHub storage.
