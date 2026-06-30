# Week 7 V2: Rollback-Constrained Unlearning

This experiment follows the completed Week 7 v1 adaptive controller without
overwriting it. V1 remains under:

`Week 7/results/adaptive_constrained_unlearning_v1`

V2 writes only to:

`Week 7/results/rollback_constrained_unlearning_v2`

## Why V2 Exists

V1 increased forgetting pressure until a utility floor was crossed, then tried
to repair the already-damaged model. Its selected checkpoint improved on Week
6, but later checkpoints showed that reducing pressure did not restore lost
retain behavior.

V2 turns the utility checks into hard acceptance rules. Each trial uses only 25
forget examples, equivalent to one quarter of the 100-example forget epoch.
After the trial:

- accept the adapter only when retain, general, and lab-number retain guards
  all pass;
- increase forget pressure by only 10% after acceptance;
- reject and delete violating updates;
- reload the last feasible adapter;
- reduce forgetting pressure and learning rate;
- increase retain preservation before retrying the same data block.

## Guardrails

- aggregate retain selection accuracy: `84%` or higher
- general selection accuracy: `52%` or higher
- lab-number retain selection accuracy: `75%` or higher
- forget held-out target: `45%` or lower

The `lab_number` retain examples are oversampled by 2x or 3x, depending on the
candidate, because v1's collateral damage was concentrated in that category.

## Focused Run

The default Colab run trains two rollback candidates for at most 20 trials and
16 accepted quarter-epoch blocks. A candidate stops after three consecutive
rejections or after reaching the feasible forgetting target.

Both candidate-best adapters receive the complete final evaluation. Selection
still uses the same deterministic forget/retain split introduced in Week 6,
plus the deterministic 25-example general guard split.

## Resume And Storage

Trial metrics, decisions, predictions, controller state, and reports are
committed to the v2 branch. Replaceable accepted/best adapters are stored under
the separate `week7-v2-rollback-resume-state` GitHub Release tag. Final
candidate and selected adapters are committed once after evaluation.

Run with:

```bash
python "Week 7/rollback_constrained_unlearning_v2/train_week7_rollback_constrained_unlearning_v2.py"
```
