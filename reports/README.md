# Weekly Project Reports

These reports reconstruct the work completed in the Unlearning Thesis project
from the notebooks, datasets, manifests, saved outputs, result tables, model
artifacts, and dated analysis files currently present in the workspace.

The workspace does not contain Git history, so dates should be interpreted as
artifact creation or preservation dates rather than a complete activity log.
The numbered "weeks" are the project's research phases, not necessarily exact
Monday-to-Sunday calendar weeks.

## Reports

1. [Week 1 - Colab LLM Pipeline](week-01-report.md)
2. [Week 2 - Synthetic Facts Dataset](week-02-report.md)
3. [Week 3 - LoRA Baselines and General-Capability Preservation](week-03-report.md)
4. [Week 3.5 - High-Accuracy Learned Baseline](week-03-5-report.md)
5. [Week 4 - Joint Training and Gradient-Ascent Unlearning](week-04-report.md)

## Project Progression

| Phase | Main question | Main outcome |
|---|---|---|
| Week 1 | Can a small instruction model run reproducibly in Colab? | A working Qwen 0.5B generation and output-saving pipeline. |
| Week 2 | Can targeted knowledge be represented in a controlled dataset? | A 500-fact synthetic dataset with identity-level forget/retain splits and 1,500 evaluation prompts. |
| Week 3 | Can LoRA learn the facts while preserving ordinary behavior? | High synthetic accuracy was achieved; always-on general capability degraded, while routing passed all three 85% targets. |
| Week 3.5 | Can we establish a clean, reproducible learned starting model? | A Qwen 0.5B LoRA baseline reached about 95% synthetic accuracy, including strong held-out paraphrase performance. |
| Week 4 | Can gradient ascent selectively suppress the forget facts? | Forget accuracy fell from 95% to 35%, but retain accuracy also fell from 94.58% to 73%. |

## Overall Research Arc

The project moved from infrastructure and dataset construction to controlled
learning, preservation experiments, and finally unlearning. The central finding
is that learning the synthetic facts is straightforward, but selective removal
without collateral damage is substantially harder. The strongest Week 4 result
supports partial selective suppression with a measurable utility trade-off, not
complete deletion of the targeted knowledge.
