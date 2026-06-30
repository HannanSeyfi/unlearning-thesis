# Week 7 V2 Results

The rollback-constrained Colab run writes its new outputs here. This directory
is separate from `../adaptive_constrained_unlearning_v1/`; the v1 results are
never overwritten.

Expected generated content includes:

- per-trial decisions, guard metrics, and selection predictions;
- resumable controller metadata;
- candidate-best and selected adapters;
- full candidate evaluations and cross-week comparisons;
- `metrics.json` and `week7_v2_rollback_report.md`.

Rolling accepted/best adapters are also backed up as replaceable assets under
the `week7-v2-rollback-resume-state` GitHub Release tag. The final adapters and
all tabular/report outputs are committed to the branch by the notebook.
