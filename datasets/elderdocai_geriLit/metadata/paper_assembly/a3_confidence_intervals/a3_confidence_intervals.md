# A3 - Confidence Intervals (frozen results)

## Experiment 1 (n=121)

| metric | mean | sd | 95% CI (t) | 95% CI (bootstrap) |
|--------|------|----|-------------|--------------------|
| faithfulness | 0.7541 | 0.3966 | [0.6835, 0.8248] | [0.6818, 0.8223] |
| answer_relevance | 0.6715 | 0.4496 | [0.5914, 0.7516] | [0.5930, 0.7500] |
| evidence_support | 0.8528 | 0.2408 | [0.8099, 0.8957] | [0.8091, 0.8934] |

Unsupported-claim rate: 23.14% (95% binomial CI [15.96%, 31.68%])

## Experiment 3 (n=20, exact binomial 95% CI)

- state_response_rate: 7/20 = 35.00% (95% CI [15.39%, 59.22%])
- appropriateness_rate: 0/20 = 0.00% (95% CI [0.00%, 16.84%])

## Experiment 4 (n=121 paired deltas, VANILLA - FULL)

| metric | mean delta | sd | 95% CI (t) |
|--------|------------|----|------------|
| faithfulness | -0.1942 | 0.5015 | [-0.2836, -0.1048] |
| answer_relevance | -0.1405 | 0.5009 | [-0.2298, -0.0512] |
| evidence_support | -0.1785 | 0.2655 | [-0.2258, -0.1312] |