"""Frozen feature bank and support/validation indices for each experimental split."""
import re

import numpy as np


def normalize_rows(array, dtype):
    a = np.asarray(array, dtype=dtype)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Feature and text arrays must be finite, nonempty matrices.")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid embedding norm.")
    return a / norms


def split_key(shot, seed):
    return f"k{shot}_s{seed}"


def available_splits(bank):
    conditions = []
    for name in bank:
        match = re.fullmatch(r"train_k([0-9]+)_s([0-9]+)", name)
        if match is not None:
            conditions.append(tuple(map(int, match.groups())))
    if not conditions:
        raise ValueError("No train_kK_sS indices found.")
    return sorted(conditions)


def validate(bank):
    x = normalize_rows(bank["image_features"], np.float32)
    text = normalize_rows(bank["text_features"], np.float64)
    y = bank["labels"]
    names = bank["class_names"]
    if x.shape[1] != text.shape[1] or len(names) != len(text) or len(set(names.tolist())) != len(names):
        raise ValueError("Embedding dimensions or class order metadata are inconsistent.")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= len(text)):
        raise ValueError("Labels must be integer class-row indices, starting at zero.")
    if "image_ids" in bank:
        ids = bank["image_ids"]
        if ids.shape != y.shape or len(set(ids.tolist())) != len(ids):
            raise ValueError("Image IDs must identify distinct feature rows.")
    scale = float(bank["logit_scale"].item())
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("logit_scale must be the positive, already exponentiated CLIP scale.")
    return x, y, text, scale


def split_indices(bank, shot, seed, labels, nclasses):
    key = split_key(shot, seed)
    indices = [bank[f"train_{key}"], bank[f"val_{key}"], bank["test_idx"]]
    for a in indices:
        if (a.ndim != 1 or not a.size or a.dtype.kind not in "iu"
                or a.min() < 0 or a.max() >= len(labels)):
            raise ValueError(f"Invalid index array in {key}.")
    joined = np.concatenate(indices)
    if np.unique(joined).size != len(joined):
        raise ValueError(f"Duplicate or overlapping support, validation or test indices in {key}.")
    tr, va, te = indices
    if not np.all(np.bincount(labels[tr], minlength=nclasses) == shot):
        raise ValueError(f"Each class must have exactly {shot} support examples in {key}.")
    val_counts = np.bincount(labels[va], minlength=nclasses)
    if val_counts.min() < 1 or val_counts.max() > min(shot, 4):
        raise ValueError(f"Expected 1 to min(K,4) validation examples per class in {key}.")
    return tr, va, te


def demo_bank():
    rng = np.random.default_rng(6704)
    c, d = 6, 20
    centers = normalize_rows(rng.normal(size=(c, d)), np.float64)
    text = normalize_rows(centers + .24 * rng.normal(size=(c, d)), np.float64)
    per_class = 70
    labels = np.repeat(np.arange(c), per_class)
    rows = np.arange(len(labels)).reshape(c, per_class)
    bank = dict(dataset=np.array("synthetic_demo"), class_names=np.array([f"class{i}" for i in range(c)]),
                logit_scale=np.array(8.), labels=labels, text_features=text,
                image_features=centers[labels] + .30 * rng.normal(size=(len(labels), d)),
                test_idx=rows[:, 50:].ravel())
    for k in (1, 4, 16):
        for seed in (1, 2, 3):
            split_rng = np.random.default_rng(seed)
            key = split_key(k, seed)
            bank[f"train_{key}"] = np.concatenate([split_rng.permutation(row[:40])[:k] for row in rows])
            bank[f"val_{key}"] = np.concatenate([split_rng.permutation(row[40:50])[:min(k,4)] for row in rows])
    return bank
