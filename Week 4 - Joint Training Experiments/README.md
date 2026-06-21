# Preserved Experiment: Joint General and Synthetic Training

This folder contains the earlier work that was previously called Week 4. The
thesis sequence now uses Week 3.5 for the learned Qwen 0.5B baseline and Week 4
for gradient-ascent unlearning.

## Notebooks

- `week4_create_joint_training_data.ipynb`: verifies the fixed 200-question
  general dataset and creates the reduced 700-example synthetic dataset.
- `week4_train_joint_general_synthetic.ipynb`: trains and evaluates one
  always-enabled LoRA adapter on both datasets. The V2 experiment uses
  Qwen2.5-1.5B-Instruct in 4-bit, rank-16 LoRA across attention and MLP
  projections, balanced general replay, and checkpoint selection that rejects
  adapters which fail to learn the synthetic facts.

## General Dataset

- 200 unique questions
- 40 questions per category
- 140 train, 20 validation, 40 final test
- The training notebook evaluates and saves predictions for all 200 questions
  before and after training.
- The 40-question test score is the held-out generalization result.
- The post-training all-200 score is descriptive because it includes 140 seen
  training questions.

The scorer normalizes accents and number words, so answers such as `Bogotá`
versus `Bogota` and `four` versus `4` are scored consistently.

## Synthetic Dataset

- 100 fictional identities
- 500 unique underlying facts
- 700 total training examples
- 500 original prompts plus 200 additional training paraphrases
- 800 held-out evaluation paraphrases
- 160 forget and 640 retain evaluation questions

All JSONL data files are protected by SHA-256 hashes in their manifests.

## V2 Training Run

The revised notebook writes to:

`Week 4 - Joint Training Experiments/results/joint_lora_700synthetic_general_v2`

In Colab, that path is inside `/content/unlearning-thesis`, and the notebook
pushes the output folder back to GitHub when it finishes.

It does not overwrite the failed V1 run. Model selection evaluates epochs 8,
12, 16, and 20 at several adapter strengths. A checkpoint must achieve at
least 80% on the synthetic selection prompts before it is eligible, and the
final 800 synthetic paraphrases plus 40 general test questions remain held out
until selection is complete.
