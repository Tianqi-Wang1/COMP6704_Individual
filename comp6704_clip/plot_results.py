"""Rebuild figures and aggregate CSVs using completed run outputs."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def export_rows(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render(folder):
    summary = read_rows(folder / "summary.csv")
    trace = read_rows(folder / "trace.csv")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    dataset = config["dataset"]
    bench = config["args"]["benchmark_alpha"]
    conditions = sorted({(int(row["shots"]), int(row["seed"])) for row in summary})
    k, seed = next(((k,s) for k,s in conditions if k == 4), conditions[0])
    fig, ax = plt.subplots(figsize=(6.2,3.9), constrained_layout=True)
    for method in ("PGD","APG","ADMM"):
        rows = [r for r in trace if r["basis"] == "text" and int(r["shots"]) == k and int(r["seed"]) == seed
                and r["solver"] == method and float(r["alpha"]) == bench]
        ax.semilogy([float(r["seconds"]) for r in rows],
                    [max(1e-15,float(r["gap_upper"])) for r in rows], label=method)
    ax.axhline(config["args"]["tol"], color=".5", linestyle=":", label="Stopping tolerance")
    ax.set(title=f"{dataset} · {k}-shot · seed {seed} · α={bench:g}", xlabel="Solver time (s)",
           ylabel="Upper bound on objective error")
    ax.legend(frameon=False)
    fig.savefig(folder / "convergence.png", dpi=170)
    plt.close(fig)

    sweep = [r for r in summary if r["solver"] == "APG"]
    groups = defaultdict(list)
    selections = defaultdict(list)
    for row in sweep:
        groups[row["basis"], int(row["shots"]), float(row["alpha"])].append(row)
        selections[row["basis"], int(row["shots"]), int(row["seed"])].append(row)
    means = []
    for (basis, shot, alpha), rows in sorted(groups.items()):
        scores = [float(r["test_accuracy"]) for r in rows if int(r["converged"])]
        complete = len(scores) == len(rows)
        means.append(dict(basis=basis, shots=shot, alpha=alpha, n_total=len(rows), n_converged=len(scores),
                          test_mean_pct=np.mean(scores) if complete else "",
                          test_sample_sd_pp=np.std(scores,ddof=1) if complete and len(scores)>1 else ""))
    export_rows(folder/"accuracy_by_shot.csv", list(means[0]), means)
    selected = []
    for (basis, shot, selected_seed), rows in sorted(selections.items()):
        if all(int(r["converged"]) for r in rows):
            best = min(rows, key=lambda r: (-float(r["val_accuracy"]),float(r["alpha"])))
            selected.append(dict(basis=basis,shots=shot,seed=selected_seed,alpha=best["alpha"],
                                 val_accuracy=best["val_accuracy"],test_accuracy=best["test_accuracy"]))
    export_rows(folder/"validation_selected.csv",
                ["basis","shots","seed","alpha","val_accuracy","test_accuracy"], selected)
    bases = sorted({r["basis"] for r in means}, key=lambda name: name != "text")
    fig, axes = plt.subplots(1, len(bases), squeeze=False, sharey=True, figsize=(4.8 * len(bases), 3.8))
    for ax, basis in zip(axes[0], bases):
        shots = sorted({r["shots"] for r in means if r["basis"] == basis})
        for shot in shots:
            rows = [r for r in means if r["basis"] == basis and r["shots"] == shot]
            alphas = np.array([r["alpha"] for r in rows])
            avg = np.array([r["test_mean_pct"] if r["test_mean_pct"] != "" else np.nan for r in rows])
            sd = np.array([r["test_sample_sd_pp"] if r["test_sample_sd_pp"] != "" else np.nan for r in rows])
            line, = ax.plot(alphas, avg, marker="o", label=f"{shot}-shot")
            ax.fill_between(alphas, avg - sd, avg + sd, alpha=.12, color=line.get_color())
        label = "Initial classifier" if config["args"]["demo"] else "Zero-shot CLIP"
        ax.axhline(float(summary[0]["zero_shot_test_accuracy"]), color=".45", linestyle=":", label=label)
        ax.set(title="Text directions" if basis == "text" else "Random directions",
               xlabel="Relative subspace radius α", xticks=alphas)
    axes[0, 0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=min(4, len(handles)), fontsize=8)
    fig.suptitle(dataset)
    fig.tight_layout(rect=(0, .075, 1, .98))
    fig.savefig(folder / "accuracy_vs_radius.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    render(parser.parse_args().run)
