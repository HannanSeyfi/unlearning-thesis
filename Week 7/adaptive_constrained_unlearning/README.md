# Adaptive Constraint Runner

`train_week7_adaptive_constrained_unlearning.py` is the canonical Week 7
runner. It imports the stable Week 6 data, model-loading, strict-scoring, and
report helpers so the comparison protocol remains unchanged.

## Controller

At epoch `t`, the training objective is:

```text
-p_t * forget_loss
+ (retain_weight + lambda_t) * retain_loss
+ (kl_weight + dual_kl_ratio * lambda_t) * retain_kl
```

Here, `p_t` is the current forget pressure and `lambda_t` is a non-negative
preservation dual variable. After held-out selection evaluation:

- `p_t` grows if retain/general constraints pass and forgetting is above the
  target;
- `p_t` backs off if either constraint fails;
- `lambda_t` follows the signed maximum constraint gap and is clipped to a
  configured range.

The fixed-pressure control uses the same scalar training objective but disables
both updates.

## Selection And Final Evaluation

Checkpoint selection uses the deterministic Week 6 forget/retain held-out
subsets and a deterministic 25-example general guard subset. Global eligibility
requires retain accuracy of at least `82%` and general accuracy of at least
`50%`.

After the sweep, the runner fully evaluates:

- the globally selected candidate;
- the best adaptive candidate;
- the best fixed-pressure control.

Duplicate roles reuse one evaluation. Final metrics also include synthetic and
general results with checkpoint-selection examples excluded.

## Resume And GitHub Sync

Each epoch overwrites a local rolling adapter and records the next controller
state in `latest.json`. The rolling latest and candidate-best adapters are
uploaded as replaceable assets on the `week7-adaptive-resume-state` GitHub
release. Their small JSON metadata and all selection evidence are committed to
the run branch. The optimizer is intentionally restarted on resume, matching
the Week 6 resume policy.

With `--push-each-epoch`, the runner updates those assets and commits the
lightweight Week 7 progress files after every epoch. It commits the final
candidate and selected adapters once after evaluation. The Colab notebook
enables this by default.
