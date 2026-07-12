# Week 7 V3: Normalized-Gradient Rollback

V3 is a separate follow-up to the completed adaptive v1 and rollback v2 runs.
It does not overwrite either result.

## Change From V2

V2 combined scalar-weighted losses. Because forget cross-entropy was already
very small, its gradient was overwhelmed by retain and KL preservation. V3
computes forget-ascent and preservation gradients separately and scales the
forget direction to a controlled ratio of the preservation gradient norm.

Each proposal trains on 50 forget examples. A proposal is accepted only when:

- aggregate retain, general, and lab-number retain satisfy baseline-relative
  floors; and
- forget selection accuracy improves by at least 1.25 percentage points over
  the last accepted checkpoint.

Rejected proposals are rolled back. Utility violations reduce intensity;
utility-safe proposals with no forgetting increase intensity before retrying
the same data block.

## Focused Candidates

- normalized, conflict-projected forget gradients;
- normalized, unprojected stronger forget gradients.

The primary target is 55% forget held-out accuracy while retaining the derived
utility floors. The original 45% target remains a stretch target.

## Storage

Results are written only to:

`Week 7/results/normalized_rollback_unlearning_v3`

Rolling adapters use the separate
`week7-v3-normalized-resume-state` GitHub Release tag. Final candidate adapters,
metrics, gradient diagnostics, predictions, and the generated report are
committed to the v3 branch.

Run with:

```bash
python "Week 7/normalized_rollback_unlearning_v3/train_week7_normalized_rollback_unlearning_v3.py"
```
