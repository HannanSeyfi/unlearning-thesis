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
6. [Weeks 4–7 - Forgetting–Utility Trade-off Synthesis](week4-week7-thesis-synthesis.md)

## Cross-Week Synthesis Artifacts

- [Weeks 4–7 thesis synthesis](week4-week7-thesis-synthesis.md)
- [Machine-readable master comparison](week4-week7-master-comparison.csv)
- [Forget–retain trade-off figure](figures/week4-week7-forget-retain-tradeoff.svg)

## Project Progression

| Phase | Main question | Main outcome |
|---|---|---|
| Week 1 | Can a small instruction model run reproducibly in Colab? | A working Qwen 0.5B generation and output-saving pipeline. |
| Week 2 | Can targeted knowledge be represented in a controlled dataset? | A 500-fact synthetic dataset with identity-level forget/retain splits and 1,500 evaluation prompts. |
| Week 3 | Can LoRA learn the facts while preserving ordinary behavior? | High synthetic accuracy was achieved; always-on general capability degraded, while routing passed all three 85% targets. |
| Week 3.5 | Can we establish a clean, reproducible learned starting model? | A Qwen 0.5B LoRA baseline reached about 95% synthetic accuracy, including strong held-out paraphrase performance. |
| Week 4 | Can gradient ascent selectively suppress the forget facts? | Forget accuracy fell from 95% to 35%, but retain accuracy also fell from 94.58% to 73%. |
| Week 5 | Can retain and KL regularization preserve utility during unlearning? | The selected checkpoint preserved 88.9% retain held-out accuracy but left forget accuracy at 67.0%; an aggressive contrast recovered strong forgetting only by collapsing utility. |
| Week 6 | Can gradient-conflict projection improve the trade-off? | PCGrad-style projection preserved the utility floors and improved forgetting modestly to 64.0%, but did not reach the target. |
| Week 7 | Can adaptive pressure, rollback, or gradient normalization satisfy the joint target? | Adaptive V1 produced the best selected utility-constrained balance at 59.0% forget and 83.1% retain; rollback V2 and normalized V3 were preservation-first negative results. |

## Overall Research Arc

The project moved from infrastructure and dataset construction to controlled
learning, preservation experiments, and finally unlearning. The central finding
is that learning the synthetic facts is straightforward, but selective removal
without collateral damage is substantially harder. The strongest Week 4 result
supports partial selective suppression with a measurable utility trade-off, not
complete deletion of the targeted knowledge.

Weeks 5–7 make this conclusion more precise. Preservation-aware optimization
recovers much of the retained utility lost in Week 4, but none of the evaluated
checkpoints simultaneously satisfies the final forget, retain, and general
targets. The negative rollback results help identify strict preservation as a
binding constraint rather than an unqualified improvement.
