"""Benchmark three solvers, then evaluate a prespecified constraint-radius sweep."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from data import available_splits, demo_bank, split_indices, validate
from plot_results import render
from problem import Problem, accuracy, text_basis
from solvers import solve


def write_csv(path, rows):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, suffix=".csv", delete=False) as f:
        tmp = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path, help="One dataset feature-bank NPZ")
    source.add_argument("--demo", action="store_true", help="Use a seeded synthetic dataset")
    p.add_argument("--shots", nargs="+", type=int)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--reg", type=float, default=1., help="Positive L2 penalty on D")
    p.add_argument("--rho", type=float, default=1., help="ADMM penalty")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.,.25,.5,1.])
    p.add_argument("--benchmark-alpha", type=float, default=.5)
    p.add_argument("--random-control", action="store_true", help="Also sweep a matched-rank random subspace")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("runs"), help="Parent directory for new run directories")
    a = p.parse_args()
    if (not all(np.isfinite(v) and v > 0 for v in (a.tol, a.reg, a.rho))
            or min(a.rank, a.max_iter, a.log_every, a.threads) < 1
            or not all(0 <= t <= 1 for t in [*a.alphas, a.benchmark_alpha])):
        p.error("Penalties, tolerances and counts must be positive; radius fractions must be in [0, 1].")
    return a


def run(a):
    if a.demo:
        bank = demo_bank()
    else:
        with np.load(a.cache, allow_pickle=False) as src:
            bank = {key: src[key] for key in src.files}
    x, y, text, scale = validate(bank)
    dataset = str(bank["dataset"].item())
    available = available_splits(bank)
    conditions = [(k,s) for k,s in available if (a.shots is None or k in a.shots)
                  and (a.seeds is None or s in a.seeds)]
    if not conditions:
        raise ValueError(f"No matching splits. Available (shots, seed): {available}")
    if a.shots and a.seeds:
        missing = {(k,s) for k in a.shots for s in a.seeds} - set(conditions)
        if missing:
            raise ValueError(f"Missing requested splits: {sorted(missing)}")
    if a.shots and set(a.shots) - {k for k,s in conditions}:
        raise ValueError(f"Requested shot unavailable. Available: {available}")
    if a.seeds and set(a.seeds) - {s for k,s in conditions}:
        raise ValueError(f"Requested seed unavailable. Available: {available}")
    u = text_basis(text, a.rank)
    indices = {(k,s):split_indices(bank,k,s,y,len(text)) for k,s in conditions}
    alphas = sorted(set([*a.alphas, a.benchmark_alpha]))
    a.out.mkdir(parents=True, exist_ok=True)
    # Use the input filename only as metadata. Run directories have unique, safe names.
    prefix = ("demo-" if a.demo else "clip-") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=a.out))
    config = dict(args=vars(a), dataset=dataset, conditions=conditions,
                  cache_sha256=None if a.demo else file_sha256(a.cache),
                  source={k:str(bank[k].item()) for k in ("backbone","model_id","template","preprocess") if k in bank},
                  logit_scale=scale, numpy=np.__version__, scipy=scipy.__version__,
                  python=platform.python_version(), platform=platform.platform(), blas=threadpool_info())
    (folder / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    print("Output:", folder.resolve(), flush=True)
    summary, history = [], []
    for shot, seed in conditions:
        tr, va, te = indices[shot, seed]
        p = Problem(x[tr], y[tr], text, scale, a.reg)
        free = p.free_solution()
        zero_acc = accuracy(x[te], y[te], text.T)
        bases = {"text": u}
        if a.random_control:
            rng = np.random.default_rng(seed + 6704)
            bases["random"] = np.linalg.qr(rng.normal(size=u.shape), mode="reduced")[0]
        for basis_name, basis in bases.items():
            free_radius = np.linalg.norm(basis.T @ free)
            for alpha in alphas:
                radius = alpha * free_radius
                methods = (["PGD","APG","ADMM"] if basis_name == "text" and alpha == a.benchmark_alpha
                           else ["APG"])
                group_history, group_summary = [], []
                for method in methods:
                    d, trace, converged = solve(p, basis, radius, method, a.tol, a.max_iter, a.log_every, a.rho)
                    key = dict(dataset=dataset, shots=shot, seed=seed, basis=basis_name, rank=a.rank,
                               alpha=alpha, epsilon=radius, free_radius=free_radius, solver=method)
                    group_history.extend({**key,**point} for point in trace)
                    weights = text.T + d
                    group_summary.append(dict(**key, **trace[-1], converged=int(converged),
                                              val_accuracy=accuracy(x[va], y[va], weights),
                                              test_accuracy=accuracy(x[te], y[te], weights),
                                              zero_shot_test_accuracy=zero_acc))
                upper = min(row["objective"] for row in group_history)
                lower = max(row["objective"]-row["gap_upper"] for row in group_history)
                for row in group_history + group_summary:
                    row["gap_to_best_feasible"] = max(0., row["objective"] - upper)
                    row["reference_interval_width"] = max(0., upper - lower)
                history.extend(group_history)
                summary.extend(group_summary)
        write_csv(folder / "trace.csv", history)
        write_csv(folder / "summary.csv", summary)
        print(f"{dataset}: finished {shot}-shot, seed {seed}", flush=True)
    render(folder)
    capped = sum(not row["converged"] for row in summary)
    print(f"{len(summary)} fits, {capped} reached the iteration cap. Results: {folder.resolve()}", flush=True)


if __name__ == "__main__":
    args = parse()
    with threadpool_limits(limits=args.threads, user_api="blas"):
        run(args)
