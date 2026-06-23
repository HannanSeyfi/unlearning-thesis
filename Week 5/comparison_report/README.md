# Week 5 Comparison Report

This folder contains a Colab-ready report generator for comparing the Week 4
gradient-ascent unlearning result with the Week 5 retain-regularized sweep.

The report is meant to turn the Week 5 result into a thesis-ready analysis
artifact rather than only a single headline score.

## Run In Colab

Open:

`week5_comparison_report_colab.ipynb`

Colab link for this branch:

https://colab.research.google.com/github/HannanSeyfi/unlearning-thesis/blob/codex/week5-comparison-report/Week%205/comparison_report/week5_comparison_report_colab.ipynb

The notebook clones the GitHub repository, runs the report script, and displays
the generated tables and plots.

If this branch has not been merged yet, the notebook uses:

`codex/week5-comparison-report`

After merging, change `BRANCH` in the first code cell to `main`.

## Run Locally

From the repository root:

```bash
python "Week 5/comparison_report/create_week5_comparison_report.py"
```

Generated files are written to:

`Week 5/comparison_report/report_outputs`

## Outputs

- `report_outputs/week4_week5_final_metrics.csv`
- `report_outputs/week5_candidate_best_summary_ranked.csv`
- `report_outputs/week4_week5_category_breakdown.csv`
- `report_outputs/week4_week5_final_metrics.png`
- `report_outputs/week5_forget_retain_tradeoff.png`
- `report_outputs/week5_candidate_tradeoff_paths.png`
- `report_outputs/week4_week5_category_breakdown.png`
- `report_outputs/week5_comparison_report.md`
