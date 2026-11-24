# Recomputed report tables

Generated from the released summary.csv files.

## Table 2: Solver comparison

Success counts use all 27 instances. Costs use the common instances solved by all three methods.
Speedups are medians of paired runtime ratios, not ratios of median runtimes.

| Solver | Solved | Median seconds | Median gradient evaluations | Paired median speedup |
| --- | ---: | ---: | ---: | ---: |
| PGD | 18/27 | 2.375 | 8878 | 1.00x |
| APG | 27/27 | 0.260 | 871 | 8.99x |
| ADMM | 27/27 | 0.423 | 531.5 | 5.54x |

## Table 3: Test accuracy (%)

Mean +/- sample SD over three seeds; validation-selected text-subspace APG.

| Dataset | Zero-shot CLIP | 1-shot | 4-shot | 16-shot |
| --- | ---: | ---: | ---: | ---: |
| EuroSAT | 42.21 | 64.91 +/- 8.30 | 78.87 +/- 0.88 | 85.69 +/- 0.73 |
| DTD | 44.39 | 53.13 +/- 0.95 | 61.51 +/- 1.26 | 69.78 +/- 0.43 |
| Oxford-IIIT Pets | 88.20 | 89.41 +/- 0.52 | 91.52 +/- 0.86 | 92.73 +/- 0.18 |

## Table 4: Accuracy differences (percentage points)

Both text and random radii are selected independently on validation accuracy, with smaller-radius tie-breaking.
The unconstrained reference uses text-subspace alpha=1. Differences are computed before rounding.

| Dataset | Shots | Text minus unconstrained | Text minus random |
| --- | ---: | ---: | ---: |
| EuroSAT | 1 | +0.17 | +0.21 |
| EuroSAT | 4 | -0.06 | 0.00 |
| EuroSAT | 16 | -0.03 | +0.02 |
| DTD | 1 | -0.12 | -0.16 |
| DTD | 4 | -0.06 | 0.00 |
| DTD | 16 | -0.20 | -0.22 |
| Oxford-IIIT Pets | 1 | -0.01 | +0.01 |
| Oxford-IIIT Pets | 4 | -0.09 | -0.05 |
| Oxford-IIIT Pets | 16 | -0.12 | -0.14 |
