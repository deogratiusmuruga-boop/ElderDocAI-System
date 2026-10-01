# B4/B5 - v1.1 failure-mode diagnostics and failure examples

Population: 121 questions (GeriLit-Gold v1.1, frozen).

## B4a - Failure taxonomy

- Unsupported (faithfulness <= 0.5): 28 (23.14%)
- Contradiction (faithfulness == 0.0): 22 (18.18%)
- Partial low (0 < faithfulness <= 0.5): 6
- Mean final reliability of failing questions: 0.7862
- Gold chunk present in evidence for failures: 0/28

### By gate decision

| decision | n | failures | failure rate % |
|----------|---|----------|----------------|
| ACCEPT | 37 | 9 | 24.3 |
| REFINE | 84 | 19 | 22.6 |
| RE-RETRIEVE | 0 | 0 | None |
| REJECT | 0 | 0 | None |

### By topic

| topic | n | failures | failure rate % |
|-------|---|----------|----------------|
| C01 | 11 | 1 | 9.1 |
| C02 | 15 | 3 | 20.0 |
| C03 | 13 | 6 | 46.2 |
| C04 | 13 | 2 | 15.4 |
| C05 | 14 | 2 | 14.3 |
| C06 | 16 | 3 | 18.8 |
| C07 | 14 | 4 | 28.6 |
| C08 | 9 | 1 | 11.1 |
| C09 | 10 | 3 | 30.0 |
| C10 | 6 | 3 | 50.0 |

## B4b - Reliability gate and generation quality

| decision | n | mean faith | mean relv | mean support | halluc % | mean init rel | mean final rel |
|----------|---|------------|-----------|--------------|---------|-----------------|----------------|
| ACCEPT | 37 | 0.723 | 0.763 | 0.887 | 24.3 | 0.830 | 0.830 |
| REFINE | 84 | 0.768 | 0.631 | 0.838 | 22.6 | 0.770 | 0.770 |

- Spearman(reliability, faithfulness): 0.0607
- Failure rate with gold chunk present: 0.0% (33 base)
- Failure rate with gold chunk absent: 31.8%

## B5 - Failure examples (all 28 unsupported rows, deterministic sort)

| question id | topic | decision | faith | relv | support | gold |
|-------------|-------|----------|-------|------|---------|------|
| GLG11-019 | C02 | REFINE | 0.00 | 0.00 | 0.167 | False |
| GLG11-023 | C02 | ACCEPT | 0.00 | 0.00 | 1.000 | False |
| GLG11-027 | C03 | REFINE | 0.00 | 0.00 | 0.167 | False |
| GLG11-033 | C03 | REFINE | 0.00 | 0.00 | 0.000 | False |
| GLG11-051 | C04 | REFINE | 0.00 | 0.00 | 0.333 | False |
| GLG11-052 | C04 | REFINE | 0.00 | 0.00 | 0.000 | False |
| GLG11-075 | C06 | REFINE | 0.00 | 0.00 | 0.167 | False |
| GLG11-079 | C06 | REFINE | 0.00 | 0.00 | 0.778 | False |
| GLG11-082 | C06 | REFINE | 0.00 | 0.00 | 0.167 | False |
| GLG11-084 | C07 | ACCEPT | 0.00 | 0.00 | 0.556 | False |
| GLG11-089 | C07 | ACCEPT | 0.00 | 0.00 | 0.000 | False |
| GLG11-091 | C07 | REFINE | 0.00 | 0.00 | 1.000 | False |
| GLG11-092 | C07 | REFINE | 0.00 | 0.00 | 0.643 | False |
| GLG11-107 | C09 | ACCEPT | 0.00 | 0.00 | 1.000 | False |
| GLG11-110 | C09 | REFINE | 0.00 | 0.00 | 1.000 | False |
| GLG11-011 | C01 | ACCEPT | 0.00 | 1.00 | 1.000 | False |
| GLG11-036 | C03 | REFINE | 0.00 | 1.00 | 0.900 | False |
| GLG11-061 | C05 | REFINE | 0.00 | 1.00 | 0.778 | False |
| GLG11-063 | C05 | ACCEPT | 0.00 | 1.00 | 0.750 | False |
| GLG11-111 | C09 | ACCEPT | 0.00 | 1.00 | 1.000 | False |
| GLG11-118 | C10 | ACCEPT | 0.00 | 1.00 | 0.944 | False |
| GLG11-120 | C10 | REFINE | 0.00 | 1.00 | 0.667 | False |
| GLG11-032 | C03 | REFINE | 0.25 | 0.00 | 1.000 | False |
| GLG11-012 | C02 | REFINE | 0.25 | 1.00 | 0.778 | False |
| GLG11-029 | C03 | REFINE | 0.25 | 1.00 | 1.000 | False |
| GLG11-030 | C03 | REFINE | 0.25 | 1.00 | 0.818 | False |
| GLG11-102 | C08 | ACCEPT | 0.25 | 1.00 | 0.636 | False |
| GLG11-117 | C10 | REFINE | 0.25 | 1.00 | 0.273 | False |

Illustrative excerpt (worst-faithfulness row):

**Q:** According to the article, what is the association between cognitive and meta analysis based data participants people consuming high in older adults?

**A:** I couldn't find that information in the knowledge base.