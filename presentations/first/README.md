# First professor presentation

This folder contains a reproducible, CPU-only reporting notebook for the first
professor-facing thesis presentation.

[Open the notebook in Google Colab](https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/main/presentations/first/generate_unlearning_presentation.ipynb)

## Run

1. Open `generate_unlearning_presentation.ipynb` in Colab.
2. Run all cells. No GPU is required.
3. Download `results/unlearning_presentation_bundle.zip`, or use the optional
   final cell to commit regenerated results after adding `GITHUB_TOKEN` to
   Colab Secrets.

The notebook reads the versioned Week 4-7 metrics, histories, and prediction
files. It does not rerun model training and does not invent unavailable results.

## Generated outputs

Everything is written below `results/`:

- `figures/`: six presentation-ready charts in PNG and PDF formats;
- `tables/`: exact plotting data, sample sizes, Wilson intervals, candidate
  summaries, dataset audit, and qualitative before/after examples;
- `reports/professor_progress_presentation.pdf`: an eight-page presentation;
- `reports/unlearning_progress_report.pdf`: a fuller progress report;
- `reports/unlearning_progress_report.html` and `.md`: editable report forms;
- `reports/speaker_notes.md`: slide-by-slide talking points;
- `reports/oracle_retraining_protocol.md`: the missing retain-only baseline plan;
- `reports/experiment_checklist.md`: evidence required for the next milestone;
- `manifest.json`: input and output hashes;
- `unlearning_presentation_bundle.zip`: one downloadable package.

The base-model reference row comes from the thesis progress summary and is
corroborated by the repository progress reports. All unlearning-method results
come from `reports/week4-week7-master-comparison.csv` and the corresponding
versioned result folders.
