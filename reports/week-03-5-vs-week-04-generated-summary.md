# Week 3.5 vs Week 4 Result Summary

Generated from saved result files.

## Research Flow

Week 3.5 established that the model had learned the synthetic facts. Week 4 then applied gradient-ascent unlearning to reduce recall of the forget facts.

## Week 3.5 Learning Result

| Evaluation group | Examples | Base before training | LoRA after training |
|---|---|---|---|
| Forget, all prompts | 300 | 0.00% | 95.00% |
| Retain, all prompts | 1200 | 0.08% | 94.67% |
| Forget held-out paraphrases | 200 | 0.00% | 92.50% |
| Retain held-out paraphrases | 800 | 0.00% | 92.00% |
| Forget seen prompts | 100 | 0.00% | 100.00% |
| Retain seen prompts | 400 | 0.25% | 100.00% |
| General controls | 50 | 84.00% | 62.00% |

## Week 4 Unlearning Result

| Metric | Before unlearning | After unlearning | Change |
|---|---|---|---|
| Forget accuracy, all | 95.00% | 35.00% | -60.00 pp |
| Forget held-out paraphrases | 92.50% | 34.00% | -58.50 pp |
| Retain accuracy, all | 94.58% | 73.00% | -21.58 pp |
| Retain held-out paraphrases | 91.88% | 66.88% | -25.00 pp |
| General controls | 56.00% | 50.00% | -6.00 pp |

## Main Takeaway

The model learned the synthetic facts very well in Week 3.5. Week 4 then reduced forget accuracy by -60.00 pp, but retain accuracy also changed by -21.58 pp. This means the experiment demonstrates partial unlearning with a clear utility cost.
