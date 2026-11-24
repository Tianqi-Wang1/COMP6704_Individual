# Constrained Optimization for Few-Shot CLIP Classifier Adaptation

Code and experimental results for the COMP6704 individual project by **Tianqi Wang**.

This project learns an additive correction to the text classifier of a frozen CLIP model. It compares **projected gradient descent (PGD)**, **accelerated projected gradient (APG/FISTA)** and **ADMM** on the same strongly convex constrained problem, then evaluates how the constraint affects few-shot classification.

- [Project report](report/COMP6704_Report_individual.pdf)
- [Recomputed report tables](results/aggregate/REPORT_TABLES.md)
- [Convergence figure (PDF)](report/figures/COMP6704_Solver_Convergence.pdf)

## What is included?

The repository includes the optimization code, the report, and the completed **270-fit experiment** on EuroSAT, DTD and Oxford-IIIT Pets. Raw summaries and convergence traces are included, so the reported tables can be recomputed without downloading datasets or running CLIP.

| Task | Required resources |
| --- | --- |
| Recompute tables from the released results | Python standard library only |
| Run the synthetic optimization example | Python and `comp6704_clip/requirements.txt`; CPU only |
| Rerun optimization on real CLIP features | The same dependencies plus compatible feature-bank `.npz` files |
| Extract features from original images | Dataset images and splits, CLIP weights, a compatible TCOC code checkout, PyTorch, torchvision, Pillow and Dassl |

**Dataset images, CLIP checkpoints, the external TCOC code and the original feature caches are not bundled.** Consequently, the repository supports independent inspection and reaggregation of the released results, but reproducing the real-image experiments from scratch requires those additional resources. See [feature preparation and cache format](docs/feature_cache.md).

## Repository contents

| Path | Description |
| --- | --- |
| `comp6704_clip/problem.py` | Objective, gradient, projection, text subspace and unconstrained reference solve |
| `comp6704_clip/solvers.py` | PGD, APG and ADMM with a shared stopping certificate |
| `comp6704_clip/data.py` | Feature-bank checks, split checks and synthetic data |
| `comp6704_clip/prepare_features.py` | Feature extraction through the external CLIP and dataset loaders |
| `comp6704_clip/run_experiment.py` | Solver benchmark and constraint-radius sweep |
| `comp6704_clip/run_formal.py` | Linux launcher for the three-dataset experiment |
| `comp6704_clip/plot_results.py` | Per-dataset plots and validation-selected results |
| `tools/summarize_results.py` | Recompute report Tables 2-4 and numerical diagnostics |
| `results/` | Released per-dataset results and aggregated tables |
| `report/` | Report PDF and the report's convergence figure |

The seven original experiment Python files and their requirements file are preserved unchanged. The reporting helper and repository documentation were added for this release.

## Quick start

Run the following commands from the repository root. Use an existing environment or create a virtual environment first.

```bash
python -m pip install -r comp6704_clip/requirements.txt

python comp6704_clip/run_experiment.py \
  --demo --shots 4 --seeds 1 \
  --random-control --max-iter 10000 --threads 1 \
  --out runs/demo
```

This synthetic example generates its own data and runs 10 fits: four text-subspace APG settings, four random-subspace APG settings, and PGD/ADMM at text-subspace `alpha=0.5`. It writes a new timestamped directory under `runs/demo/` and prints its exact path. Synthetic accuracies are a functionality check, not the real-dataset results in the report.

The core requirements are NumPy, SciPy, Matplotlib and threadpoolctl. The original formal runs used **Python 3.9.25, NumPy 1.24.2 and SciPy 1.13.1**. Software and BLAS details are recorded in each released `config.json`.

## Recompute the report tables

This command uses only the Python standard library and does not rerun optimization:

```bash
python tools/summarize_results.py --results results --out analysis_results
```

It validates the three 90-fit grids and writes:

| Generated file | Contents |
| --- | --- |
| `REPORT_TABLES.md` | Readable versions of report Tables 2-4 |
| `table_2_solver_comparison.csv` | Solver success counts, common-instance runtime and paired speedups |
| `table_3_few_shot_accuracy.csv` | Validation-selected accuracy means and sample standard deviations |
| `table_4_constraint_comparison.csv` | Text-subspace accuracy differences against both controls |
| `validation_selected.csv` | Selected radius for every dataset/shot/seed/basis |
| `diagnostics.json` | Capped-run diagnostics, objective differences and validation ties |

The checked outputs are also committed under `results/aggregate/`. The helper expects one formal run directory per dataset and rejects incomplete or mismatched grids rather than pooling unrelated runs.

To regenerate the original plots for one dataset, pass its run directory, for example:

```bash
python comp6704_clip/plot_results.py \
  results/eurosat/clip-20261123T080429Z-84s2109r
```

This updates the derived CSVs and PNGs in that directory; `summary.csv` and `trace.csv` are the underlying records. The separate PDF in `report/figures/` is the figure used in the report, not an output of this per-dataset plotting command.

## Rerun the real-data optimization

### Using compatible cached features

Place one feature bank per dataset under `cache_formal/`, following the [documented schema](docs/feature_cache.md). For example, run EuroSAT with the exact formal solver settings:

```bash
python comp6704_clip/run_experiment.py \
  --cache cache_formal/eurosat.npz \
  --shots 1 4 16 --seeds 1 2 3 \
  --rank 4 --reg 1 --rho 1 \
  --alphas 0 0.25 0.5 1 --benchmark-alpha 0.5 \
  --random-control --tol 1e-5 --max-iter 10000 \
  --log-every 5 --threads 1 \
  --out runs/reproduction/eurosat
```

Repeat with `dtd.npz` and `oxford_pets.npz`, changing the output dataset name accordingly. Once compatible features are available, these solver runs use the CPU and do not require the external TCOC checkout or a GPU. The standalone experiment script defaults to **3,000** iterations; keep the explicit `--max-iter 10000` option when matching the report.

## Optimization problem

Let $x_i$ be a fixed normalized image feature, $W_0$ the initial normalized text classifier, and $D$ a trainable class-wise correction. The logits are

$$z_i(D)=s(W_0+D)^\top x_i.$$

We solve

$$\min_D\;\frac{1}{n}\sum_{i=1}^{n}\operatorname{CE}(z_i(D),y_i)+\frac{\lambda}{2}\lVert D\rVert_F^2
\quad\text{subject to}\quad\lVert U_r^\top D\rVert_F\leq\epsilon.$$

$U_r$ contains the leading left singular vectors of the uncentered text classifier. The radius is calibrated from a training-only regularized unconstrained solution:

$$\epsilon=\alpha\lVert U_r^\top D_{\mathrm{free}}\rVert_F.$$

The corrected weights are **not renormalized**, preserving affine logits and convexity. Positive quadratic regularization makes the objective strongly convex. Here, `--rank 4` specifies the constrained subspace dimension; it does not impose a rank-four factorization on $D$.

- **PGD:** gradient update followed by closed-form projection.
- **APG:** FISTA extrapolation with the same step size and projection.
- **ADMM:** split the projected correction into an auxiliary variable and use warm-started L-BFGS-B for the smooth subproblem.

All methods start from zero. A fit is successful only when both the certified objective suboptimality upper bound and the original iterate's constraint violation are at most $10^{-5}$. The final classifier uses the projected feasible correction.

## Experimental protocol

| Setting | Value |
| --- | --- |
| Datasets | EuroSAT (10 classes), DTD (47), Oxford-IIIT Pets (37) |
| Frozen backbone | CLIP ViT-B/16 |
| Prompt | `a photo of a {}.` |
| Logit scale in the released runs | 100 |
| Shots per class | 1, 4, 16 |
| Seeds | 1, 2, 3 |
| Validation budget | At most `min(K, 4)` examples per class |
| Subspace dimension / regularization / ADMM penalty | `r=4`, `lambda=1`, `rho=1` |
| Radius fractions | 0, 0.25, 0.5, 1 |
| Solver benchmark | Text subspace, `alpha=0.5` |
| Iteration limit / checkpoint interval | 10,000 / 5; outer iterations for ADMM |
| Optimization | CPU, double-precision arithmetic, one BLAS thread |

Each of the 27 dataset/shot/seed splits has 10 fits, giving 270 fits overall. The text and random subspaces each have an APG radius sweep. Random bases have the same dimension and are fixed across shot counts for each seed. PGD and ADMM supply the two additional fits at the benchmark radius.

For classification, radii are selected independently for each split and basis using validation accuracy; ties favor smaller radii. The test set is used only for reporting. Means and **sample** standard deviations are calculated over three seeds.

## Main results

### Solver comparison

Success counts use all 27 benchmark instances. Runtime, gradient evaluations and paired speedups use the **18 common instances** solved to tolerance by all three methods.

| Solver | Successful instances | Median time (s) | Median gradient evaluations | Paired median speedup over PGD |
| --- | ---: | ---: | ---: | ---: |
| PGD | 18/27 | 2.375 | 8,878 | 1.00x |
| APG | 27/27 | 0.260 | 871 | 8.99x |
| ADMM | 27/27 | 0.423 | 531.5 | 5.54x |

Speedups are the median of per-instance runtime ratios, not the ratio of median times. All nine capped runs are PGD at 16 shots. On the 18 commonly solved instances, the maximum within-instance objective spread is $7.30\times10^{-6}$ and the maximum test-accuracy spread is 0.18 percentage points.

Timing includes solver updates, ADMM inner solves and stopping checks. It excludes feature extraction, SVD, step-bound computation, unconstrained calibration, final accuracy evaluation and plotting. These are solver speedups, not end-to-end adaptation speedups. Each released fit was timed once; the reported medians summarize different problem instances rather than repeated timing trials of a single instance.

### Few-shot classification

Validation-selected text-subspace APG, mean +/- sample standard deviation of test accuracy (%):

| Dataset | Zero-shot CLIP | 1-shot | 4-shot | 16-shot |
| --- | ---: | ---: | ---: | ---: |
| EuroSAT | 42.21 | 64.91 +/- 8.30 | 78.87 +/- 0.88 | 85.69 +/- 0.73 |
| DTD | 44.39 | 53.13 +/- 0.95 | 61.51 +/- 1.26 | 69.78 +/- 0.43 |
| Oxford-IIIT Pets | 88.20 | 89.41 +/- 0.52 | 91.52 +/- 0.86 | 92.73 +/- 0.18 |

Classifier correction improves mean accuracy in every evaluated dataset/shot setting. The extra text-subspace constraint provides no consistent classification advantage under the tested configuration: averaged equally across the nine dataset/shot combinations, its accuracy differences from the unconstrained and random-subspace references are -0.06 and -0.04 percentage points. The study evaluates this optimization formulation, not a state-of-the-art comparison against other adaptation methods.

## Reproducibility notes

- The released result CSVs and original plots are unchanged. Machine-specific absolute filesystem paths in `config.json` were normalized for publication; numerical settings and cache hashes were retained.
- Dataset images, split files and cached features are separate resources. Seeds alone do not guarantee identical support sets across different dataset loaders or pre-existing few-shot caches.
- Match image splits, class order, prompts, feature preprocessing and CLIP checkpoint when reproducing the real-data results. The exporter reads the checkpoint's exponentiated logit scale; it was 100 in these formal runs.
- The reported timings depend on the original hardware and numerical libraries. No CPU model was recorded in the provided result metadata.
- The evaluated scope is three datasets, one backbone and fixed subspace/regularization settings. The conclusions about the additional geometric constraint are limited to this configuration.

Background and references are provided in the report.
