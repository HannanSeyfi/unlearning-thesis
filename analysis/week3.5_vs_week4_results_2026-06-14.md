# Week 3.5 Learning vs Week 4 Unlearning

Analysis date: 2026-06-14

## Experiment summary

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Week 3.5: LoRA training on 500 synthetic fact examples
- Week 4: gradient ascent on forget loss plus gradient descent on retain loss
- Week 4 selected checkpoint: epoch 3 of 8
- Evaluation: 300 forget prompts, 1,200 retain prompts, and 50 general controls

## Main results

| Metric | Before unlearning | After unlearning | Change |
|---|---:|---:|---:|
| Forget accuracy, all | 95.00% | 35.00% | -60.00 pp |
| Forget accuracy, held-out paraphrases | 92.50% | 34.00% | -58.50 pp |
| Forget accuracy, training-identical prompts | 100.00% | 37.00% | -63.00 pp |
| Retain accuracy, all | 94.58% | 73.00% | -21.58 pp |
| Retain accuracy, held-out paraphrases | 91.88% | 66.88% | -25.00 pp |
| Retain accuracy, training-identical prompts | 100.00% | 85.25% | -14.75 pp |
| General control | 56.00% | 50.00% | -6.00 pp |

The method produces substantial forgetting, including on held-out paraphrases, but the
forget set remains 35% recoverable. The cost to retained synthetic knowledge is also
large: 264 previously correct retain predictions become incorrect.

## Category effects

| Fact category | Forget before | Forget after | Retain before | Retain after |
|---|---:|---:|---:|---:|
| Access phrase | 100.00% | 61.67% | 100.00% | 90.42% |
| Favorite city | 100.00% | 71.67% | 100.00% | 97.92% |
| Lab number | 83.33% | 11.67% | 77.92% | 49.17% |
| Research topic | 100.00% | 15.00% | 97.08% | 64.17% |
| Secret code | 91.67% | 15.00% | 97.92% | 63.33% |

Unlearning is strongly category-dependent. Structured identifiers, research topics,
and secret codes are suppressed most, but they also suffer the largest retain damage.
Favorite cities and access phrases preserve retain performance well, but are not
forgotten effectively.

## What the wrong answers mean

Of the 195 incorrect post-unlearning forget answers:

- 0 are empty.
- 0 are refusal-style answers.
- 88 exactly equal another value in the synthetic dataset.
- 85 equal a value belonging to the retain set.
- 79 are substitutions from the same fact category.

This is evidence of fact substitution or interference, not clean erasure. Accuracy
reduction alone should therefore not be described as complete deletion of the targeted
knowledge.

Forgetting also varies greatly by identity. Three of the 20 forget identities reach 0%
accuracy, while one remains at 73.3%. Aggregate accuracy hides this unevenness.

## Checkpoint choice

Epoch 3 was selected because its training-sample forget accuracy was 37% while retain
sample accuracy remained 92%. Later epochs reduced forget accuracy as low as 7%, but
retain sample accuracy fell to 60-76%, below the experiment's 85% eligibility threshold.

The selected checkpoint is a reasonable tradeoff under that threshold, but it does not
meet the configured target of at most 20% forget accuracy.

## Evaluation caveats

1. Week 3.5 reported general-control `contains_value` using substring matching. This
   incorrectly counted outputs such as `560` for expected `56` and `120` for expected
   `12`. With Week 4's boundary-aware matching, the comparable pre-unlearning general
   score is 56%, not the originally reported 62%.
2. Week 4 reproduced 1,497 of 1,500 Week 3.5 synthetic prediction texts exactly before
   unlearning. The three differences caused a one-example retain-score difference
   (94.67% versus 94.58%) and are consistent with small 4-bit inference variation.
3. General controls contain only 50 examples, so a single question changes the score by
   2 percentage points.
4. The experiment measures output suppression, not parameter-level deletion or
   resistance to extraction attacks. Stronger prompts, relearning tests, and
   membership/knowledge probing are still needed for a defensible unlearning claim.

## Conclusion

Week 4 demonstrates partial, generalized unlearning: forget accuracy falls by 60 points
and the reduction transfers to held-out paraphrases. However, the method has substantial
collateral damage, remains vulnerable on 35% of forget prompts, and often replaces the
target with another memorized synthetic fact. The strongest thesis claim supported by
this run is selective suppression with a measurable utility tradeoff, not complete
machine unlearning.
