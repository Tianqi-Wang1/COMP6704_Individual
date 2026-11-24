"""Rebuild the report tables from the released three-dataset CSVs (standard library only)."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median, stdev


DATASETS = ("eurosat", "dtd", "oxford_pets")
NAMES = {"eurosat": "EuroSAT", "dtd": "DTD", "oxford_pets": "Oxford-IIIT Pets"}
SHOTS = (1, 4, 16)
SEEDS = (1, 2, 3)
ALPHAS = (0.0, 0.25, 0.5, 1.0)
METHODS = ("PGD", "APG", "ADMM")
INT_FIELDS = {"shots", "seed", "rank", "iteration", "grad_evals", "converged"}
STRING_FIELDS = {"dataset", "basis", "solver"}
EXPECTED = {(k, s, b, a, m) for k in SHOTS for s in SEEDS
            for b in ("text", "random") for a in ALPHAS
            for m in (METHODS if b == "text" and a == 0.5 else ("APG",))}


def read_results(root):
    rows = []
    for dataset in DATASETS:
        files = sorted((root / dataset).glob("clip-*/summary.csv"))
        if len(files) != 1:
            raise ValueError(f"Expected one released summary.csv for {dataset}; found {len(files)}")
        config = json.loads(files[0].with_name("config.json").read_text(encoding="utf-8"))
        expected_args = dict(rank=4, reg=1.0, rho=1.0, benchmark_alpha=0.5,
                             tol=1e-5, max_iter=10000, log_every=5,
                             threads=1, random_control=True)
        if config["dataset"] != dataset or any(config["args"].get(k) != v for k, v in expected_args.items()):
            raise ValueError(f"Configuration does not match the report: {files[0].parent}")
        with files[0].open(newline="", encoding="utf-8") as f:
            dataset_rows = []
            for raw in csv.DictReader(f):
                r = {k: v if k in STRING_FIELDS else int(v) if k in INT_FIELDS else float(v)
                     for k, v in raw.items()}
                if any(not math.isfinite(v) for v in r.values() if isinstance(v, (int, float))):
                    raise ValueError(f"Nonfinite value in {files[0]}")
                if r["dataset"] != dataset or r["rank"] != 4:
                    raise ValueError(f"Unexpected dataset or subspace dimension in {files[0]}")
                certified = r["gap_upper"] <= 1e-5 and r["violation"] <= 1e-5
                if r["converged"] not in (0, 1) or bool(r["converged"]) != certified:
                    raise ValueError(f"Convergence flag and certificate disagree in {files[0]}")
                dataset_rows.append(r)
        identities = {(r["shots"], r["seed"], r["basis"], r["alpha"], r["solver"])
                      for r in dataset_rows}
        if len(dataset_rows) != 90 or identities != EXPECTED:
            raise ValueError(f"Incomplete or duplicate experimental grid: {files[0]}")
        rows.extend(dataset_rows)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def signed_pp(value):
    return "0.00" if abs(value) < 0.005 else f"{value:+.2f}"


def summarize(rows, out):
    benchmark = defaultdict(dict)
    scans = defaultdict(list)
    for r in rows:
        split = (r["dataset"], r["shots"], r["seed"])
        if r["basis"] == "text" and r["alpha"] == 0.5:
            benchmark[split][r["solver"]] = r
        if r["solver"] == "APG":
            scans[(*split, r["basis"])].append(r)
    common = {k: g for k, g in benchmark.items() if all(r["converged"] for r in g.values())}
    if not common:
        raise ValueError("No instances were solved by all three methods")
    if not all(r["converged"] for rs in scans.values() for r in rs):
        raise ValueError("This report aggregation requires all APG radius sweeps to reach tolerance")

    solver_table = []
    for method in METHODS:
        solver_table.append(dict(
            solver=method,
            successful_instances=sum(g[method]["converged"] for g in benchmark.values()),
            benchmark_instances=len(benchmark), common_instances=len(common),
            median_seconds=median(g[method]["seconds"] for g in common.values()),
            median_grad_evals=median(g[method]["grad_evals"] for g in common.values()),
            paired_median_speedup=median(g["PGD"]["seconds"] / g[method]["seconds"]
                                        for g in common.values())))

    selected = {k: min(rs, key=lambda r: (-r["val_accuracy"], r["alpha"])) for k, rs in scans.items()}
    selection_rows = [{k: r[k] for k in ("dataset", "shots", "seed", "basis", "alpha",
                                       "val_accuracy", "test_accuracy")}
                      for _, r in sorted(selected.items())]
    accuracy_table, difference_table, alpha_spreads = [], [], []
    for dataset in DATASETS:
        for shot in SHOTS:
            text = [selected[(dataset, shot, s, "text")]["test_accuracy"] for s in SEEDS]
            random = [selected[(dataset, shot, s, "random")]["test_accuracy"] for s in SEEDS]
            free = [next(r["test_accuracy"] for r in scans[(dataset, shot, s, "text")]
                         if r["alpha"] == 1.0) for s in SEEDS]
            zero_values = {r["zero_shot_test_accuracy"] for r in rows if r["dataset"] == dataset}
            if len(zero_values) != 1:
                raise ValueError(f"Zero-shot baseline varies within {dataset}")
            zero = next(iter(zero_values))
            accuracy_table.append(dict(dataset=dataset, shots=shot, seeds=len(SEEDS),
                                       zero_shot_accuracy_pct=zero,
                                       test_mean_pct=mean(text), test_sample_sd_pp=stdev(text),
                                       gain_over_zero_shot_pp=mean(text)-zero))
            difference_table.append(dict(dataset=dataset, shots=shot,
                                         text_minus_unconstrained_pp=mean(text)-mean(free),
                                         text_minus_random_pp=mean(text)-mean(random)))
            alpha_means = [mean(next(r["test_accuracy"] for r in scans[(dataset, shot, s, "text")]
                                    if r["alpha"] == a) for s in SEEDS) for a in ALPHAS]
            alpha_spreads.append(max(alpha_means)-min(alpha_means))

    failed = [g["PGD"] for g in benchmark.values() if not g["PGD"]["converged"]]
    fastest = Counter(min((r for r in g.values() if r["converged"]),
                          key=lambda r: r["seconds"])["solver"] for g in benchmark.values())
    diagnostics = dict(
        total_fits=len(rows), benchmark_instances=len(benchmark), common_instances=len(common),
        capped_fits=sum(not r["converged"] for r in rows),
        capped_fits_by_dataset={d: sum(not r["converged"] for r in rows if r["dataset"] == d)
                               for d in DATASETS},
        fastest_to_tolerance=dict(fastest), apg_scan_fits=sum(len(rs) for rs in scans.values()),
        max_within_instance_objective_spread=max(max(r["objective"] for r in g.values())-
                                                min(r["objective"] for r in g.values())
                                                for g in common.values()),
        max_within_instance_test_spread_pp=max(max(r["test_accuracy"] for r in g.values())-
                                              min(r["test_accuracy"] for r in g.values())
                                              for g in common.values()),
        pgd_capped_shots=sorted({r["shots"] for r in failed}),
        pgd_capped_gap_min=min((r["gap_upper"] for r in failed), default=None),
        pgd_capped_gap_max=max((r["gap_upper"] for r in failed), default=None),
        pgd_capped_max_violation=max((r["violation"] for r in failed), default=None),
        all_alpha_validation_ties={b: sum(len({r["val_accuracy"] for r in rs}) == 1
                                          for k, rs in scans.items() if k[3] == b)
                                   for b in ("text", "random")},
        text_selected_alpha_counts=dict(Counter(str(r["alpha"]) for k, r in selected.items()
                                                if k[3] == "text")),
        text_objective_nonincrease_count=sum(
            all(a["objective"] >= b["objective"] for a, b in zip(
                sorted(rs, key=lambda r: r["alpha"]), sorted(rs, key=lambda r: r["alpha"])[1:]))
            for k, rs in scans.items() if k[3] == "text"),
        max_alpha_mean_accuracy_spread_pp=max(alpha_spreads),
        macro_text_minus_unconstrained_pp=mean(r["text_minus_unconstrained_pp"] for r in difference_table),
        macro_text_minus_random_pp=mean(r["text_minus_random_pp"] for r in difference_table))

    out.mkdir(parents=True, exist_ok=True)
    for filename, table in [("table_2_solver_comparison.csv", solver_table),
                            ("table_3_few_shot_accuracy.csv", accuracy_table),
                            ("table_4_constraint_comparison.csv", difference_table),
                            ("validation_selected.csv", selection_rows)]:
        write_csv(out / filename, table)
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2)+"\n", encoding="utf-8")

    lines = ["# Recomputed report tables", "", "Generated from the released summary.csv files.", "",
             "## Table 2: Solver comparison", "",
             "Success counts use all 27 instances. Costs use the common instances solved by all three methods.",
             "Speedups are medians of paired runtime ratios, not ratios of median runtimes.", "",
             "| Solver | Solved | Median seconds | Median gradient evaluations | Paired median speedup |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in solver_table:
        lines.append(f"| {r['solver']} | {r['successful_instances']}/{r['benchmark_instances']} | "
                     f"{r['median_seconds']:.3f} | {r['median_grad_evals']:g} | {r['paired_median_speedup']:.2f}x |")
    lines.extend(["", "## Table 3: Test accuracy (%)", "", "Mean +/- sample SD over three seeds; validation-selected text-subspace APG.", "",
                  "| Dataset | Zero-shot CLIP | 1-shot | 4-shot | 16-shot |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for dataset in DATASETS:
        rs = [r for r in accuracy_table if r["dataset"] == dataset]
        values = " | ".join(f"{r['test_mean_pct']:.2f} +/- {r['test_sample_sd_pp']:.2f}" for r in rs)
        lines.append(f"| {NAMES[dataset]} | {rs[0]['zero_shot_accuracy_pct']:.2f} | {values} |")
    lines.extend(["", "## Table 4: Accuracy differences (percentage points)", "",
                  "Both text and random radii are selected independently on validation accuracy, with smaller-radius tie-breaking.",
                  "The unconstrained reference uses text-subspace alpha=1. Differences are computed before rounding.", "",
                  "| Dataset | Shots | Text minus unconstrained | Text minus random |",
                  "| --- | ---: | ---: | ---: |"])
    for r in difference_table:
        lines.append(f"| {NAMES[r['dataset']]} | {r['shots']} | "
                     f"{signed_pp(r['text_minus_unconstrained_pp'])} | {signed_pp(r['text_minus_random_pp'])} |")
    (out / "REPORT_TABLES.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("analysis_results"))
    args = parser.parse_args()
    d = summarize(read_results(args.results), args.out)
    print(f"Validated {d['total_fits']} fits; {d['capped_fits']} reached the iteration cap.")
    print(f"Solver comparison: {d['benchmark_instances']} instances; {d['common_instances']} commonly solved.")
    print(f"Tables and diagnostics: {args.out.resolve()}")


if __name__ == "__main__":
    main()
