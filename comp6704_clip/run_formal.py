"""Run or extend the COMP6704 benchmark on EuroSAT, DTD and Oxford-IIIT Pets (Linux)."""
import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from zipfile import BadZipFile

import numpy as np

from data import split_key
from plot_results import render
from run_experiment import file_sha256


HERE = Path(__file__).resolve().parent
DATASETS = ("eurosat", "dtd", "oxford_pets")
BACKBONE = "ViT-B/16"
TEMPLATE = "a photo of a {}."
FLAGS = dict(shots=[1, 4, 16], seeds=[1, 2, 3], rank=4, reg=1., rho=1.,
             alphas=[0., .25, .5, 1.], benchmark_alpha=.5, random_control=True,
             tol=1e-5, max_iter=10000, log_every=5, threads=1)
CONDITIONS = [(k, s) for k in FLAGS["shots"] for s in FLAGS["seeds"]]
EXPECTED = {(k, s, basis, alpha, method)
            for k, s in CONDITIONS for basis in ("text", "random") for alpha in FLAGS["alphas"]
            for method in (("PGD", "APG", "ADMM") if basis == "text" and alpha == FLAGS["benchmark_alpha"]
                           else ("APG",))}


def choose_batch(out):
    if out is not None:
        return out.expanduser().resolve()
    runs = Path.cwd() / "runs"
    previous = sorted(p for p in runs.glob("formal_20??????_??????") if p.is_dir())
    return previous[-1] if previous else runs / datetime.now().strftime("formal_%Y%m%d_%H%M%S")


def legacy_job_active(folder):
    # The original two-dataset nohup command directs both parent and children here.
    log = folder / "launcher.log"
    if not log.is_file():
        return False
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc.name != str(os.getpid()) and proc.stat().st_uid == os.getuid()
                    and (proc / "fd/1").samefile(log)):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def compatible_cache(path, name):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as bank:
            keys = set(bank.files)
            required = {"labels", "image_features", "class_names", "text_features", "test_idx", "logit_scale"}
            required.update(f"{part}_{split_key(k,s)}" for k,s in CONDITIONS for part in ("train", "val"))
            return (required <= keys and bank["dataset"].item() == name
                    and bank["template"].item() == TEMPLATE and bank["backbone"].item() == BACKBONE)
    except (KeyError, ValueError, OSError, BadZipFile, EOFError):
        return False


def ensure_cache(a, name):
    root = a.cache_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    filename = name + ".npz"
    candidates = [root / filename, *sorted(root.glob(name + "-formal-*/" + filename), reverse=True)]
    for path in candidates:
        if compatible_cache(path, name):
            print(f"Reusing features: {path}", flush=True)
            return path
    dest = root if not (root / filename).exists() else Path(tempfile.mkdtemp(prefix=name+"-formal-", dir=root))
    env = os.environ.copy()
    if a.gpu is not None:
        uuid = subprocess.check_output(["nvidia-smi", "-i", a.gpu, "--query-gpu=uuid",
                                        "--format=csv,noheader"], text=True).strip()
        env["CUDA_VISIBLE_DEVICES"] = uuid
    print(f"Extracting features for {name}.", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / "prepare_features.py"),
                    "--tcoc-root", str(a.tcoc_root), "--data-root", str(a.data_root),
                    "--datasets", name, "--shots", *map(str, FLAGS["shots"]), "--seeds", *map(str, FLAGS["seeds"]),
                    "--backbone", BACKBONE, "--template", TEMPLATE, "--device", "cuda",
                    "--cache-dir", str(dest)], env=env, check=True)
    return dest / filename


def completed(parent, name, checksum):
    for path in sorted(parent.glob("clip-*/config.json"), reverse=True):
        run = path.parent
        if not (run / "summary.csv").is_file() or not (run / "trace.csv").is_file():
            continue
        try:
            cfg = json.loads(path.read_text())
            if (cfg["dataset"] != name or cfg["cache_sha256"] != checksum
                    or any(cfg["args"].get(k) != v for k,v in FLAGS.items())):
                continue
            with (run / "summary.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            keys = {(int(r["shots"]), int(r["seed"]), r["basis"], float(r["alpha"]), r["solver"]) for r in rows}
            if (keys != EXPECTED or len(rows) != len(EXPECTED)
                    or any(r["dataset"] != name or r["converged"] not in ("0","1") for r in rows)):
                continue
        except (ValueError, KeyError, OSError, csv.Error):
            continue
        if any(not (run / f).is_file() for f in ("accuracy_by_shot.csv", "validation_selected.csv",
                                                 "convergence.png", "accuracy_vs_radius.png")):
            render(run)
        return run, sum(r["converged"] == "0" for r in rows)
    return None


def run_batch(a, folder):
    if legacy_job_active(folder):
        print("Waiting for the active EuroSAT/DTD job to finish. Its log is " + str(folder / "launcher.log"), flush=True)
        while legacy_job_active(folder):
            time.sleep(5)
    results = []
    solver_args = []
    for key, value in FLAGS.items():
        solver_args.append("--" + key.replace("_", "-"))
        if value is not True:
            solver_args.extend(map(str, value if isinstance(value, list) else [value]))
    for name in DATASETS:
        cache = ensure_cache(a, name)
        checksum = file_sha256(cache)
        parent = folder / name
        found = completed(parent, name, checksum)
        if found is None:
            print(f"Starting {name}: {len(EXPECTED)} fits.", flush=True)
            subprocess.run([sys.executable, "-u", str(HERE / "run_experiment.py"), "--cache", str(cache),
                            "--out", str(parent), *solver_args], check=True)
            found = completed(parent, name, checksum)
            if found is None:
                raise RuntimeError(f"No complete formal results found for {name} under {parent}.")
        else:
            print(f"Reusing completed solver results for {name}.", flush=True)
        run, capped = found
        print(f"{name}: {len(EXPECTED)} fits, {capped} reached the iteration cap. {run}", flush=True)
        results.append((name, run))
    target = folder / "formal_results_3datasets.tar.gz"
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".tar.gz", delete=False) as f:
        temp = Path(f.name)
    try:
        with tarfile.open(temp, "w:gz") as archive:
            for name, run in results:
                archive.add(run, arcname=name + "/" + run.name)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Combined results: {len(EXPECTED)*len(results)} fits. {target}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tcoc-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("cache_formal"))
    p.add_argument("--out", type=Path, help="Defaults to the latest timestamped runs/formal_* batch, or creates one.")
    p.add_argument("--gpu", help="Physical GPU index or UUID for feature extraction. Defaults to the visible device.")
    args = p.parse_args()
    batch = choose_batch(args.out)
    batch.mkdir(parents=True, exist_ok=True)
    print(f"Formal batch: {batch}", flush=True)
    with (batch / ".three_dataset.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"A three-dataset launcher is already running in {batch}.")
        run_batch(args, batch)
