# Week 4 Report: Joint Training and Gradient-Ascent Unlearning

## 1. Scope and Chronology

Two bodies of work are associated with Week 4:

1. an earlier joint general-and-synthetic LoRA training branch;
2. the final thesis-sequence Week 4 gradient-ascent unlearning experiment.

The earlier branch was preserved under
`Week 4 - Joint Training Experiments`. The active `Week 4` folder was then
redefined as the first direct unlearning stage using the Week 3.5 adapter.

## 2. Earlier Joint-Training Branch

### Purpose

The joint-training work tested whether a single always-enabled adapter could
learn synthetic facts while retaining broad question-answering ability through
balanced training.

### General dataset

The versioned `general_knowledge_200_v1` dataset contained:

- 200 unique questions;
- 5 categories;
- 40 questions per category;
- 140 training questions;
- 20 validation questions;
- 40 final test questions.

Categories were arithmetic, geography, science, language, and common knowledge.

### Reduced synthetic dataset

The versioned `synthetic_examples_700_v2` dataset contained:

- 100 fictional identities;
- 500 unique facts;
- 700 training examples;
- 500 original prompts;
- 200 extra training paraphrases;
- 800 fully held-out evaluation paraphrases;
- 160 forget evaluation questions;
- 640 retain evaluation questions;
- zero prompt overlap between training and evaluation.

### Training design

The V2 experiment used:

- `Qwen2.5-1.5B-Instruct`;
- 4-bit loading;
- rank-16 LoRA;
- attention and MLP projection modules;
- balanced general replay;
- checkpoint evaluation at epochs 8, 12, 16, and 20;
- multiple adapter strengths;
- an 80% synthetic selection threshold.

The final 800 synthetic paraphrases and 40 general test questions were held out
until model selection was complete.

### Outcome status

The workspace contains the notebooks, datasets, manifests, and corrected V2
packages, but it does not contain a final V2 result table in the main preserved
folder. The documentation refers to a failed V1 run and a corrected V2 design.
Accordingly, this branch should be reported as an experiment design and
preserved attempt, not as a demonstrated successful result.

## 3. Gradient-Ascent Unlearning Experiment

### Research question

Can the high-accuracy Week 3.5 adapter be modified so it stops producing the
designated forget facts while retaining the remaining synthetic knowledge and
ordinary behavior?

### Starting point

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Source adapter: Week 3.5 high-accuracy LoRA
- Forget evaluation prompts: 300
- Retain evaluation prompts: 1,200
- General controls: 50

Only LoRA adapter parameters were updated.

### Optimization objective

The experiment combined:

- gradient ascent on forget loss, to make the forget answers less likely;
- gradient descent on retain loss, to protect retained knowledge.

The objective was:

`objective = -forget_loss + retain_weight * retain_loss`

Configuration:

- maximum epochs: 8;
- learning rate: `5e-5`;
- retain weight: `1.0`;
- batch size: 2;
- gradient accumulation steps: 4.

### Evaluation isolation

The 1,500 synthetic evaluation rows and 50 general controls were not used for
parameter updates or checkpoint selection. Training-identical and held-out
paraphrase results were recorded separately.

## 4. Checkpoint Selection

Training-sample diagnostics developed as follows:

| Epoch | Forget accuracy | Retain sample accuracy | Eligible |
|---:|---:|---:|---|
| 1 | 82% | 100% | Yes |
| 2 | 62% | 97% | Yes |
| 3 | 37% | 92% | Yes |
| 4 | 25% | 74% | No |
| 5 | 20% | 76% | No |
| 6 | 15% | 60% | No |
| 7 | 9% | 68% | No |
| 8 | 7% | 73% | No |

Epoch 3 was selected because it produced the strongest forgetting among
checkpoints that maintained at least 85% retain-sample accuracy.

Later epochs forgot more aggressively, but their retain damage made them
ineligible. This exposed the core trade-off directly.

## 5. Final Results

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Forget accuracy, all | 95.00% | 35.00% | -60.00 pp |
| Forget held-out paraphrases | 92.50% | 34.00% | -58.50 pp |
| Forget training-identical prompts | 100.00% | 37.00% | -63.00 pp |
| Retain accuracy, all | 94.58% | 73.00% | -21.58 pp |
| Retain held-out paraphrases | 91.88% | 66.88% | -25.00 pp |
| Retain training-identical prompts | 100.00% | 85.25% | -14.75 pp |
| General controls | 56.00% | 50.00% | -6.00 pp |

The experiment produced substantial generalized forgetting: held-out forget
accuracy fell by 58.5 percentage points. However, it did not reach the
configured target of at most 20% forget accuracy, and retained knowledge
suffered significant collateral damage.

## 6. Category-Level Effects

| Category | Forget before | Forget after | Retain before | Retain after |
|---|---:|---:|---:|---:|
| Access phrase | 100.00% | 61.67% | 100.00% | 90.42% |
| Favorite city | 100.00% | 71.67% | 100.00% | 97.92% |
| Lab number | 83.33% | 11.67% | 77.92% | 49.17% |
| Research topic | 100.00% | 15.00% | 97.08% | 64.17% |
| Secret code | 91.67% | 15.00% | 97.92% | 63.33% |

Structured identifiers, research topics, and secret codes were suppressed most
strongly, but these categories also experienced the largest retain damage.
Favorite cities and access phrases retained well but were not forgotten
effectively.

## 7. Error Analysis

There were 195 incorrect post-unlearning forget answers.

- None were empty.
- None were refusal-style answers.
- 88 exactly matched another value from the synthetic dataset.
- 85 matched a value belonging to the retain set.
- 79 were substitutions from the same fact category.

This indicates interference or fact substitution rather than clean erasure.
The model often returned a different memorized synthetic value instead of
refusing or becoming uncertain.

Forgetting was also uneven across identities:

- 3 of 20 forget identities reached 0% accuracy;
- one forget identity remained at 73.3% accuracy.

Aggregate accuracy therefore concealed substantial variation.

## 8. Scoring and Reproducibility Findings

Week 4 introduced boundary-aware matching for short values. This corrected
false positives from Week 3.5 substring scoring. The comparable pre-unlearning
general score was 56%, not the originally reported 62%.

Week 4 reproduced 1,497 of 1,500 Week 3.5 prediction texts exactly before
unlearning. The three differences changed the retain score by one example and
were consistent with small 4-bit inference variation.

## 9. Main Conclusions

1. Gradient ascent substantially reduced recall of the targeted facts.
2. The reduction generalized to held-out paraphrases.
3. Forget accuracy remained too high for a claim of complete unlearning.
4. Retain accuracy fell by more than 21 percentage points overall.
5. Most wrong forget answers represented substitutions, not clean deletion.
6. Effects differed greatly by fact category and identity.
7. The defensible result is partial selective suppression with collateral
   damage, not verified parameter-level erasure.

## 10. Deliverables

- Joint-training data-creation and training notebooks
- Versioned 200-question general dataset
- Versioned 700-example synthetic dataset
- `Week 4/notebooks/week4_gradient_ascent_unlearning.ipynb`
- Unlearning history and checkpoint diagnostics
- Before/after forget, retain, and general predictions
- Selected unlearned adapter
- Metrics and percentage summaries
- Detailed June 14, 2026 comparative analysis

## 11. Limitations and Next Research Requirements

- Only one principal gradient-ascent configuration was completed.
- The 50-question general set has two-percentage-point resolution.
- Accuracy reduction does not establish parameter-level deletion.
- No extraction attack, relearning test, membership probe, or adversarial
  prompting evaluation was performed.
- Forgetting remained incomplete and uneven.
- The retain penalty was too large for strong utility-preservation claims.

Future experiments should explore lower learning rates, different retain
weights, layer or module restrictions, regularization against the source
adapter, stronger refusal-aware evaluation, relearning resistance, and
adversarial extraction tests.

## 12. Evidence Used

- `Week 4/README.md`
- `Week 4/notebooks/week4_gradient_ascent_unlearning.ipynb`
- `Week 4 - Joint Training Experiments/README.md`
- Joint-training dataset manifests
- `imports/2026-06-14-week4/Week 4/results/gradient_ascent_unlearning_v1`
- `analysis/week3.5_vs_week4_results_2026-06-14.md`
