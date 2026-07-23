# Retain-only oracle protocol

1. Start from the same Qwen2.5-0.5B-Instruct base checkpoint used for the LoRA baseline.
2. Use the same LoRA target modules, rank, optimizer, scheduler, maximum tokens, and scoring code.
3. Remove every forget-person training example; train only on the retain-person subset.
4. Select checkpoints using retain validation and general validation only. Never use forget evaluation for checkpoint selection.
5. Evaluate on the unchanged forget, retain, and general final sets.
6. Repeat with the same seeds used for the main unlearning methods.
7. Compare every unlearned model with this oracle using accuracy, token log-probability, output distributions, and paired per-question differences.

The original base model is not a sufficient oracle because it lacks both forget and retain synthetic facts.
