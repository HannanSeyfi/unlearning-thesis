# Week 1 Report: Colab LLM Pipeline

## 1. Purpose

Week 1 established the technical foundation for the thesis. The immediate goal
was to verify that a small instruction-tuned language model could be loaded and
used reliably in Google Colab before introducing fine-tuning or unlearning.

This phase reduced infrastructure risk. It tested the runtime, model-loading,
prompt formatting, deterministic generation, batch evaluation, and output-saving
steps that later experiments depended on.

## 2. Main Work Completed

### Colab runtime setup

The notebook documented how to select a T4 GPU runtime and check GPU and library
availability. This created a repeatable starting procedure for later notebooks.

### Small-model selection

The primary model was:

- `Qwen/Qwen2.5-0.5B-Instruct`

A smaller fallback was also identified:

- `HuggingFaceTB/SmolLM2-360M-Instruct`

The decision to begin with a small model prioritized short, repeatable,
affordable experiments over scale. This was appropriate for iterative thesis
work in Colab.

### Tokenizer and model loading

The pipeline loaded an instruction tokenizer and causal language model, then
used the model's chat template when available. This ensured prompts were
formatted consistently with the model's instruction-tuning format.

### Generation helper

A reusable answer-generation function was created. It supported:

- instruction-style prompts;
- deterministic generation;
- configurable output length;
- decoding only the generated continuation;
- later reuse in forget-set and retain-set evaluation.

### Thesis-oriented smoke tests

The model was asked questions related to machine unlearning, including:

- defining targeted forgetting;
- distinguishing forget and retain sets;
- naming evaluation metrics;
- explaining the value of small models for Colab experiments.

The saved Week 1 batch contains four prompt-answer rows.

### Output persistence

Generated answers were saved in structured formats. Preserved outputs include:

- `week1_generated_answers.csv`;
- `week1_generated_answers.jsonl`;
- `step_01_generation_examples.json`.

This established the research practice of saving the model identifier, prompts,
answers, and experiment outputs instead of relying only on notebook display.

## 3. Observations

The pipeline worked, but the generated answers also demonstrated why a
controlled thesis dataset was necessary. For example, Qwen sometimes described
"machine unlearning" as an AI learning from mistakes, which is not the intended
meaning in this thesis. This showed that open-ended conceptual answers were not
a reliable basis for measuring targeted forgetting.

The Week 1 output therefore served two purposes:

1. It confirmed the technical pipeline.
2. It exposed the ambiguity of evaluating unlearning through unrestricted
   natural-language explanations.

## 4. Decisions Made

- Use a small Qwen instruction model as the primary experimental model.
- Keep generation deterministic where possible for reproducibility.
- Save outputs from every experiment in machine-readable files.
- Move to synthetic factual knowledge with exact target values.
- Separate future examples into forget and retain groups.

## 5. Deliverables

- `Week 1/notebooks/week1_colab_llm_pipeline.ipynb`
- Preserved generated-answer CSV and JSONL files in the organized Drive copy
- Example generation JSON
- A reusable Colab workflow for model loading and inference

## 6. Limitations

- No training or unlearning was performed.
- The prompts were a smoke test rather than a formal benchmark.
- Only four batch prompts were preserved.
- Open-ended answers were difficult to score objectively.
- The model's incorrect conceptual descriptions reinforced the need for a
  deterministic synthetic task.

## 7. Transition to Week 2

Week 1 concluded with a concrete next step: create fictional identities and
facts, divide them into forget and retain sets, generate training and evaluation
questions, and save the dataset for later LoRA fine-tuning.

## 8. Evidence Used

- `Week 1/notebooks/week1_colab_llm_pipeline.ipynb`
- `Google Drive Upload Ready/Thesis/Week 1/results/week1_generated_answers.csv`
- `Google Drive Upload Ready/Thesis/Week 1/results/week1_generated_answers.jsonl`
- `Google Drive Upload Ready/Thesis/Week 1/results/step_01_generation_examples.json`
