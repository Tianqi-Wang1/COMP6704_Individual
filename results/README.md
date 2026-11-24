# Released experimental results

Each dataset directory contains one formal run with 90 fits: 1, 4 and 16 shots, each with seeds 1, 2 and 3. The three datasets contain 270 fits in total. The 9 runs that reached the iteration cap are retained; all are PGD runs at 16 shots.

| File | Contents |
| --- | --- |
| `config.json` | Solver arguments, feature-cache SHA-256, CLIP provenance, Python/NumPy/SciPy and BLAS information |
| `summary.csv` | One row per fit, final objective/certificate/violation, runtime, gradient counts and accuracies |
| `trace.csv` | Convergence checkpoints for all fits, including capped runs |
| `accuracy_by_shot.csv` | APG accuracy means and sample standard deviations for each fixed radius and basis |
| `validation_selected.csv` | Per-split validation-selected radius and corresponding test accuracy |
| `convergence.png` | Original per-dataset convergence plot |
| `accuracy_vs_radius.png` | Original per-dataset radius sweep plot |

The `aggregate/` directory contains report Tables 2-4 and additional diagnostics, rebuilt by `tools/summarize_results.py`. Run from the repository root:

```bash
python tools/summarize_results.py --results results --out analysis_results
```

Accuracies in the raw CSV files are percentages, not fractions. Standard deviations and accuracy differences are in percentage points. Solver runtime is measured in seconds.

All original CSVs and plots are preserved byte for byte. Only machine-specific absolute filesystem paths in `config.json` have been replaced with `<original-project>` and `<original-python-environment>` markers. These fields describe the original run and are not paths to use on another computer. Numerical settings, package versions, model identifiers and cache hashes are unchanged.

The original feature banks are not included. A cache hash identifies an original bank; it does not provide a way to download that bank.
