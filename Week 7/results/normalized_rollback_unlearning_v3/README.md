# Week 7 V3 Results

This directory contains the completed normalized-gradient rollback v3 run. The
run is independent from the completed v1, v2, and trial-8 audit directories.

## Outcome

The selected adapter is `n01_normalized_projected_balanced` at trial 13. V3 is a
preservation-first negative result rather than an improvement over Week 7 V1:
it reduced forget held-out accuracy only from 92.5% to 90.0%, while Week 7 V1
reached 59.0%. Retain held-out accuracy stayed at 91.9%, and general controls
rose from 56.0% to 60.0%.

Lower forget accuracy is better for unlearning. Higher retain and general
accuracy are better for preservation.

## Contents

- `best_week7_v3_adapter/`: selected v3 adapter with a filled model card
- `candidate_adapters/`: selected candidate adapter artifact
- `results/metrics.json`: run metadata, selected candidate, guardrails, and final evaluation
- `results/candidate_best_summary.csv`: selection-time candidate ranking
- `results/candidate_final_evaluations.csv`: final candidate evaluation metrics
- `results/week7_v3_cross_week_comparison.csv`: comparison against earlier weeks
- `results/week7_v3_normalized_report.md`: generated summary report
- `results/gradient_diagnostics.csv`: gradient norms, conflict rates, projection rates, and normalization factors
- `resume_state/`: resumable controller state and selection predictions

Rolling accepted and best adapters were also backed up under the separate
`week7-v3-normalized-resume-state` GitHub Release tag during the run.
