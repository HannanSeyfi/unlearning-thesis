# Week 2 Report: Synthetic Facts Dataset

## 1. Purpose

Week 2 converted the broad unlearning problem into a controlled factual-learning
task. The objective was to create knowledge that was fictional, measurable, and
cleanly divisible into forget and retain subsets.

This design avoided real personal data and made it possible to score whether a
model produced a specific learned value.

## 2. Dataset Design

The dataset was named `synthetic_facts_v1` and generated with seed `42`.

### Identities and facts

- 100 fictional identities
- 5 facts per identity
- 500 unique facts in total

The five fact categories were:

- secret code;
- favorite city;
- research topic;
- lab number;
- access phrase.

### Forget and retain split

The split was performed at identity level:

- 20 forget identities, representing 20% of the identities;
- 80 retain identities;
- 100 forget facts;
- 400 retain facts.

Identity-level splitting was an important methodological choice because it
prevented one fictional person from appearing in both groups.

### Training data

The dataset contains 500 training examples, one stable training prompt per
fact. Separate files were created for:

- all training examples;
- forget training examples;
- retain training examples.

### Evaluation data

The evaluation set contains 1,500 questions:

- 300 forget questions;
- 1,200 retain questions.

Each fact has three evaluation prompts:

- one prompt matching the training wording;
- two paraphrased prompts.

Therefore:

- 500 prompts are training-identical diagnostics;
- 1,000 prompts are held-out paraphrases.

This made it possible to distinguish exact prompt memorization from
generalization to alternative wording.

## 3. Main Work Completed

### Synthetic content generation

The notebook constructed fictional names and sampled values for each fact
category. Each identity was represented in both an identity table and a
long-format fact table.

### Question-answer conversion

Facts were transformed into instruction-style question-answer examples. The
training and evaluation records used chat-format `messages`, making them ready
for Hugging Face and LoRA training.

### Data validation

The notebook validated:

- expected identity and fact counts;
- uniqueness of records;
- separation of forget and retain identities;
- training and evaluation row counts;
- JSONL readability after saving.

### Structured export

The dataset was saved in both CSV and JSONL formats:

- `identities.csv`;
- `facts_long.csv`;
- `qa_train_all.csv`;
- `qa_eval_all.csv`;
- `train_all.jsonl`;
- `train_forget.jsonl`;
- `train_retain.jsonl`;
- `eval_all.jsonl`;
- `eval_forget.jsonl`;
- `eval_retain.jsonl`;
- `metadata.json`.

The preserved metadata records a UTC creation time of
`2026-06-07T14:26:38.643949+00:00`.

## 4. Methodological Value

The Week 2 dataset provided four forms of experimental control:

1. **Known provenance:** every fact was generated for the experiment.
2. **No privacy exposure:** no real person's data was used.
3. **Explicit targeting:** forget and retain membership was known exactly.
4. **Objective scoring:** expected values could be checked with exact-match or
   normalized contains-value metrics.

The paraphrase design was especially important. A model that answers only the
training-identical wording has memorized a prompt pattern; a model that answers
held-out paraphrases demonstrates broader factual learning.

## 5. Decisions Made

- Use identity-level rather than random row-level splitting.
- Reserve 20% of identities for forgetting.
- Use simple atomic values suitable for deterministic scoring.
- Preserve both original prompts and unseen paraphrases.
- Export both human-readable CSV and training-ready JSONL.
- Use this same fixed dataset across learning and unlearning stages.

## 6. Deliverables

- `Week 2/notebooks/week2_synthetic_facts_dataset.ipynb`
- Complete `Week 2/data/synthetic_facts_v1` dataset
- Dataset metadata and summary counts
- Separate all, forget, and retain JSONL files
- Evaluation data supporting seen-prompt and held-out reporting

## 7. Limitations

- Synthetic facts are much simpler than real-world knowledge.
- The task measures controlled factual recall, not broad semantic knowledge.
- Five fixed categories may not represent all memorization behaviors.
- Contains-value scoring can produce false positives without careful token or
  boundary handling.
- A successful result cannot by itself establish privacy removal in a deployed
  language model.

## 8. Transition to Week 3

With the controlled dataset complete, the next task was to train a LoRA adapter
on all 500 facts, measure learning on forget and retain questions, and examine
whether synthetic specialization damaged ordinary model behavior.

## 9. Evidence Used

- `Week 2/notebooks/week2_synthetic_facts_dataset.ipynb`
- `Week 2/data/synthetic_facts_v1/metadata.json`
- All CSV and JSONL files under `Week 2/data/synthetic_facts_v1`
