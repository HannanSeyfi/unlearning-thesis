# Week 6 Conclusions

Week 6 is a partial success. The constrained-gradient method improved the
forget/retain trade-off relative to the preserving Week 5 checkpoint, but it
did not reach the planned forgetting target.

## Selected Result

The selected checkpoint was `c02_pcgrad_balanced`, epoch 4:

| Metric | Week 5 preserving | Week 6 | Change | Week 6 target |
| --- | ---: | ---: | ---: | ---: |
| Forget held-out accuracy | 67.0% | 64.0% | -3.0 pp | 45.0% or lower |
| Retain held-out accuracy | 88.9% | 87.0% | -1.9 pp | 85.0% or higher |
| General-control accuracy | 58.0% | 54.0% | -4.0 pp | 50.0% or higher |

Lower forget accuracy is better; higher retain and general-control accuracy
are better.

The method therefore preserved both utility thresholds while producing a
small improvement in forgetting. It remained far from the intended forget
band, however, and should not be presented as solving the trade-off.

## What The Experiment Shows

- Gradient conflicts were common: projection was activated on 56% of the
  selected candidate's training steps.
- On the selection split, the projected candidate outperformed the matched
  no-projection control: 61.3% versus 67.5% forget accuracy, with 87.5% versus
  86.9% retain accuracy.
- Compared with the Week 5 aggressive checkpoint, Week 6 recovered 25.1
  percentage points of retain accuracy and 8.0 points of general accuracy,
  but surrendered 32.0 points of forgetting performance.
- Fixed PCGrad-style projection moved the Pareto frontier modestly; it did not
  create enough forgetting pressure to reach the target region.

The no-projection comparison above uses the selection split. Only the selected
Week 6 checkpoint received the complete final evaluation, so the control
should receive the same full evaluation before making a stronger causal claim
about projection.

## Decision

Keep Week 6 as evidence that gradient constraints can reduce collateral
damage, but record the forgetting target as unmet. Do not rerun the same fixed
sweep.

The next experiment should use adaptive constraint optimization: increase
forgetting pressure dynamically while enforcing explicit retain and general
performance floors. It should also fully evaluate the strongest projected and
unprojected finalists under the same protocol.

## Evidence

- [Generated Week 6 report](results/constrained_gradient_unlearning_v1/results/week6_constrained_gradient_report.md)
- [Final metrics](results/constrained_gradient_unlearning_v1/results/metrics.json)
- [Cross-week comparison](results/constrained_gradient_unlearning_v1/results/week4_week5_week6_comparison.csv)
- [Candidate ranking](results/constrained_gradient_unlearning_v1/results/candidate_best_summary.csv)
