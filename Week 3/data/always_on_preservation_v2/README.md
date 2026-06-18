# Always-On Preservation v2

This fixed dataset supports selection of a LoRA adapter that remains enabled
for both synthetic and general questions.

- `general_replay_v2`: 125 balanced general examples used in training.
- `general_validation_v2`: 40 disjoint examples used for checkpoint selection.
- `manifest.json`: record counts and SHA-256 hashes.

The separate `general_controls_v1/general_control.jsonl` file remains untouched
until final evaluation.
