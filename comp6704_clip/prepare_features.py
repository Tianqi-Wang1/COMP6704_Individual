"""Export deterministic CLIP features using the original TCOC dataset splits."""
import argparse
import hashlib
from importlib import import_module
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from data import split_indices, split_key


LOADERS = {
    "dtd": "DescribableTextures",
    "eurosat": "EuroSAT",
    "oxford_pets": "OxfordPets",
}


def collect(name, data_root, shots, seeds, datasets):
    bank = {"dataset": np.array(name)}
    paths, labels, path_to_row = [], [], {}

    def indices(items):
        rows = []
        for item in items:
            path = Path(item.impath).resolve()
            if path not in path_to_row:
                path_to_row[path] = len(paths)
                paths.append(path)
                labels.append(int(item.label))
            row = path_to_row[path]
            if labels[row] != item.label:
                raise ValueError(f"Inconsistent labels for {path}")
            rows.append(row)
        return np.asarray(rows, dtype=np.int64)

    for k in shots:
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            cfg = SimpleNamespace(SEED=seed, DATASET=SimpleNamespace(
                ROOT=str(data_root), NUM_SHOTS=k, SUBSAMPLE_CLASSES="all"))
            ds = datasets[name](cfg)
            names = np.array([ds.lab2cname[i] for i in range(len(ds.classnames))])
            te = indices(ds.test)
            if "test_idx" in bank and (not np.array_equal(te, bank["test_idx"])
                                       or not np.array_equal(names, bank["class_names"])):
                raise ValueError("Class order or test split changed across conditions.")
            bank["test_idx"], bank["class_names"] = te, names
            key = split_key(k, seed)
            bank[f"train_{key}"] = indices(ds.train_x)
            bank[f"val_{key}"] = indices(ds.val)
    bank["labels"] = np.array(labels, dtype=np.int64)
    bank["image_ids"] = np.array([Path(os.path.relpath(p, data_root)).as_posix() for p in paths])
    for k in shots:
        for seed in seeds:
            split_indices(bank, k, seed, bank["labels"], len(bank["class_names"]))
    return bank, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcoc-root", required=True, type=Path, help="Directory containing clip/ and datasets/.")
    parser.add_argument("--data-root", required=True, type=Path, help="Parent directory of the TCOC dataset folders.")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--datasets", nargs="+", choices=list(LOADERS), default=list(LOADERS))
    parser.add_argument("--shots", nargs="+", type=int, default=[1,4,16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1,2,3])
    parser.add_argument("--backbone", default="ViT-B/16", help="TCOC CLIP name or local checkpoint path")
    parser.add_argument("--template", default="a photo of a {}.")
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset cache after a complete export.")
    args = parser.parse_args()
    root = args.tcoc_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    for relative in ["clip/clip.py", *(f"datasets/{name}.py" for name in args.datasets)]:
        if not (root / relative).is_file():
            parser.error(f"Missing {relative} under --tcoc-root: {root}")
    if args.template.count("{}") != 1 or args.batch_size < 1 or min(args.shots) < 1 or min(args.seeds) < 0:
        parser.error("Check the single {} prompt placeholder, batch size, shots and seeds.")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if (args.cache_dir / f"{name}.npz").exists() and not args.overwrite:
            parser.error(f"{name}.npz already exists. Reuse it, choose another cache directory, or pass --overwrite.")
        if not (data_root / name).is_dir():
            parser.error(f"Missing dataset directory: {data_root / name}")
    sys.path.insert(0, str(root))
    import torch
    from PIL import Image
    from clip import clip
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = clip.load(args.backbone, device=device, jit=False)
    model.eval()
    classes = {name: getattr(import_module(f"datasets.{name}"), LOADERS[name]) for name in args.datasets}
    for name in args.datasets:
        bank, paths = collect(name, data_root, args.shots, args.seeds, classes)
        prompts = [args.template.format(n.replace("_", " ")) for n in bank["class_names"]]
        with torch.inference_mode():
            tokens = clip.tokenize(prompts).to(device)
            text = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=1)
            bank["text_features"] = text.cpu().numpy()
            bank["logit_scale"] = np.array(model.logit_scale.detach().float().exp().item())
            features = []
            for i in range(0, len(paths), args.batch_size):
                batch = []
                for path in paths[i:i+args.batch_size]:
                    with Image.open(path) as im:
                        batch.append(transform(im.convert("RGB")))
                z = model.encode_image(torch.stack(batch).to(device)).float()
                features.append(torch.nn.functional.normalize(z, dim=1).cpu().numpy())
            bank["image_features"] = np.concatenate(features)
        checkpoint = Path(args.backbone).expanduser()
        if checkpoint.is_file():
            with checkpoint.open("rb") as f:
                digest = hashlib.sha256()
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            model_id = digest.hexdigest()
        else:
            model_id = clip._MODELS[args.backbone]
        bank.update(backbone=np.array(args.backbone), model_id=np.array(model_id),
                    template=np.array(args.template), preprocess=np.array("CLIP evaluation resize + center crop"))
        target = args.cache_dir / f"{name}.npz"
        with tempfile.NamedTemporaryFile(dir=args.cache_dir, suffix=".npz", delete=False) as f:
            tmp = Path(f.name)
        try:
            np.savez(tmp, **bank)
            if args.overwrite:
                os.replace(tmp, target)
            else:
                os.link(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        print(f"Saved {target} ({len(paths)} distinct images, {len(prompts)} classes).", flush=True)


if __name__ == "__main__":
    main()
