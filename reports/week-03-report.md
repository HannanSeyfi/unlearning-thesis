# Week 3 Report: LoRA Baselines and Capability Preservation

## 1. Purpose

Week 3 established whether the Qwen 0.5B model could learn the Week 2 facts and
investigated the central learning trade-off: high synthetic-fact accuracy versus
preservation of ordinary language-model behavior.

The work progressed through several related experiments:

1. a clean LoRA learning baseline;
2. a balanced adapter trained with general replay;
3. checkpoint and adapter-strength selection;
4. a routed dual-target experiment;
5. an always-on preservation sweep.

## 2. Experimental Foundation

The primary model was `Qwen/Qwen2.5-0.5B-Instruct`.

The core LoRA configuration used in the successful high-capacity runs included:

- rank `16`;
- alpha `32`;
- dropout `0.05`;
- attention and MLP projection targets;
- up to 20 epochs;
- learning rates around `2e-4` to `3e-4`.

The synthetic training set contained 500 examples. Final synthetic evaluation
used 300 forget and 1,200 retain questions.

## 3. Clean LoRA Baseline

The baseline notebook evaluated a fresh base model, trained a new LoRA adapter
using only `train_all.jsonl`, and repeated the same evaluation afterward.

Key safeguards included:

- evaluation files were excluded from training;
- each evaluation row was marked as seen or held out;
- base and LoRA predictions were saved separately;
- row-level before/after comparisons were produced;
- adapter weights were archived.

This experiment demonstrated that the small model could learn the synthetic
facts very accurately. However, the later preserved high-accuracy run also
showed a clear decline in general-control performance, motivating preservation
experiments.

## 4. General-Control Framework

Week 3 introduced fixed general-capability datasets covering:

- arithmetic;
- geography;
- science;
- language;
- common knowledge.

The `general_controls_v1` design separated three roles:

| Set | Size | Purpose |
|---|---:|---|
| General control | 50 | Final held-out evaluation only |
| General replay | 30 | Ordinary-behavior examples mixed into training |
| General validation | 20 | Checkpoint and strength selection |

The final 50 controls were not used in training or model selection.

The files were versioned with SHA-256 hashes in `manifest.json`. This improved
reproducibility and prevented silent changes to evaluation data.

## 5. Balanced LoRA Experiment

The balanced run mixed:

- 500 synthetic examples;
- 30 general replay examples repeated 10 times;
- 800 mixed rows per epoch.

It trained for up to 20 epochs and evaluated checkpoints at epochs 8, 12, 16,
and 20. Adapter strengths `0.55`, `0.70`, `0.85`, and `1.00` were tested.

Candidates had to reach at least 70% on synthetic selection prompts. Eligible
candidates were compared using a harmonic balance between synthetic learning
and general validation.

The selected configuration was:

- epoch `20`;
- adapter strength `0.85`;
- synthetic selection accuracy `100%`;
- general validation accuracy `65%`;
- harmonic balance score `78.79`.

### Final balanced results

| Metric | Base | Balanced LoRA |
|---|---:|---:|
| Forget synthetic accuracy | 0.00% | 89.00% |
| Retain synthetic accuracy | 0.08% | 86.75% |
| General-control accuracy | 90.00% | 68.00% |

The balanced adapter learned both synthetic splits above 85%, but ordinary
capability fell by 22 percentage points. Replay reduced neither the preservation
problem nor the final trade-off enough to satisfy a three-way 85% target.

## 6. Routed Dual-85 Experiment

The next experiment changed the serving policy:

- synthetic questions used the trained LoRA adapter;
- ordinary questions used the same base model with the adapter disabled.

This preserved the specialized synthetic capability while avoiding adapter
interference on general questions.

The selected adapter again used epoch 20 and strength 0.85.

### Final routed results

| Metric | Result | 85% target |
|---|---:|---:|
| Synthetic forget | 89.33% | Passed |
| Synthetic retain | 88.58% | Passed |
| Routed general controls | 88.00% | Passed |
| General controls with adapter always on | 66.00% | Diagnostic only |

All three primary targets passed under routing.

This result was technically successful but conceptually qualified. General
capability was preserved because ordinary questions bypassed the adapter, not
because the adapter itself preserved the capability. The 66% always-on
diagnostic made this distinction explicit.

## 7. Always-On Preservation Sweep

An additional notebook searched for one adapter that could remain enabled for
both synthetic and ordinary questions. It varied:

- lower LoRA ranks;
- narrower target modules;
- gentler learning rates;
- earlier checkpoints;
- expanded general replay;
- joint synthetic/general validation.

The expanded `always_on_preservation_v2` data contained:

- 125 replay questions;
- 40 validation questions.

These sets were disjoint from the final 50 controls and protected by SHA-256
hashes. The notebook was designed to train and preserve every candidate and
select a global winner without using the final tests.

The workspace preserves the notebook and data design, but no final always-on
sweep result artifact is present in the main result folders. Therefore, no
unsupported claim of an 85/85/85 always-on success should be made.

## 8. Main Findings

1. Qwen 0.5B with rank-16 LoRA could learn the synthetic facts well.
2. Both forget and retain facts exceeded 85% after training.
3. Always-on specialization substantially damaged general-control accuracy.
4. Replay and adapter-strength tuning improved balance but did not eliminate
   the capability trade-off.
5. Routing restored general performance by disabling the adapter for ordinary
   questions.
6. Passing routed targets did not prove that a single always-enabled model
   retained general capability.

## 9. Research Decisions

- Preserve final controls as strictly held-out tests.
- Use separate replay, validation, and test sets.
- Evaluate multiple epochs and adapter strengths.
- Save all selection metrics and final predictions.
- Report the adapter-always-on diagnostic alongside routed performance.
- Build a cleaner high-accuracy baseline before beginning unlearning.

## 10. Deliverables

- Four Week 3 experiment notebooks
- Versioned `general_controls_v1` data
- Expanded `always_on_preservation_v2` data
- Balanced-run result tables
- Routed dual-85 result tables and routing configuration
- Saved adapters and checkpoint-selection metrics in the organized archive

## 11. Limitations

- General evaluation used only 50 final questions.
- Contains-value scoring was permissive and later required boundary-aware
  correction.
- Routing solved a deployment policy problem rather than the underlying
  always-on preservation problem.
- Synthetic selection prompts included seen training wording.
- The always-on sweep lacks a preserved completed result in the main workspace.

## 12. Transition to Week 3.5

The unlearning experiment required a clearly defined learned starting model.
Week 3.5 therefore isolated and reproduced the high-accuracy Qwen 0.5B baseline,
copied the source data and controls into a self-contained phase, and archived a
successful adapter for Week 4.

## 13. Evidence Used

- All notebooks in `Week 3/notebooks`
- `Week 3/data/general_controls_v1/manifest.json`
- `Week 3/data/always_on_preservation_v2/manifest.json`
- `Week 3/results/analysis_balanced_20260610`
- `Week 3/results/analysis_dual85_20260611`
- `Week 3/results/README.md`
