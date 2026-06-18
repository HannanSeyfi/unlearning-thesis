# General Controls v1

This directory contains the fixed Week 3 evaluation protocol.

- `general_control`: 50 final held-out questions. Never train or select on these.
- `general_replay`: 30 disjoint examples used for behavior-preservation replay.
- `general_validation`: 20 disjoint examples used for checkpoint selection.
- `manifest.json`: record counts and SHA-256 hashes for the JSONL files.

CSV files are included for human inspection. The notebooks read JSONL files and
verify their hashes before evaluation.
