# Feature preparation and cache format

The optimization code operates on cached image and text features. The repository does not include the original feature banks, dataset images or the external TCOC repository. This guide describes the inputs needed to rerun the real-image experiments.

## Extract using a compatible TCOC checkout

`prepare_features.py` expects an external directory containing at least these modules:

| Relative path | Expected interface |
| --- | --- |
| `clip/clip.py` | OpenAI-CLIP-compatible `load`, `tokenize` and `_MODELS` |
| `datasets/eurosat.py` | `EuroSAT` dataset class |
| `datasets/dtd.py` | `DescribableTextures` dataset class |
| `datasets/oxford_pets.py` | `OxfordPets` dataset class |

These dataset classes must support `SEED`, `DATASET.ROOT`, `DATASET.NUM_SHOTS` and `DATASET.SUBSAMPLE_CLASSES="all"`, and expose `train_x`, `val`, `test`, `classnames` and `lab2cname`. Each datum supplies `impath` and `label`. The referenced TCOC checkout and its dependencies are external prerequisites, not files provided by this release. Use the environment in which that checkout can already load its datasets and CLIP; the four core optimization requirements alone do not install this extraction environment.

The data root is the parent of the dataset folders:

| Dataset | Image directory used in the original setup | Split metadata used by the loader |
| --- | --- | --- |
| EuroSAT | `eurosat/2750/` | `eurosat/split_zhou_EuroSAT.json` |
| DTD | `dtd/images/` | `dtd/split_zhou_DescribableTextures.json` |
| Oxford-IIIT Pets | `oxford_pets/images/` | `oxford_pets/split_zhou_OxfordPets.json` |

The loaders may require additional original dataset metadata. Existing few-shot subsets are read from each dataset's `split_fewshot/shot_K-seed_S.pkl`. If absent, the loader may generate them. For exact support-set reproduction, supply the original splits with valid image paths; regenerated splits are not guaranteed to match across loader versions.

From the repository root:

```bash
python comp6704_clip/prepare_features.py \
  --tcoc-root /path/to/TCOC \
  --data-root /path/to/DATA \
  --datasets eurosat dtd oxford_pets \
  --shots 1 4 16 --seeds 1 2 3 \
  --backbone ViT-B/16 \
  --template 'a photo of a {}.' \
  --device cuda --cache-dir cache_formal
```

The extractor can also use `--device cpu`; feature extraction will be slower. `run_formal.py` explicitly requests CUDA when features are missing. Existing compatible feature banks can be used entirely on CPU by `run_experiment.py`.

Images use CLIP's deterministic evaluation resizing and center crop. Image and text embeddings are normalized. Class names are ordered by numeric label and underscores are replaced with spaces in prompts. The exported `logit_scale` is the exponentiated checkpoint parameter, not its logarithm. Named CLIP checkpoints may need to be downloaded by the external loader; `--backbone` can also point to a local compatible checkpoint for direct extraction.

The extractor avoids replacing an existing cache unless `--overwrite` is supplied. Each successful export saves one `.npz` bank per dataset. Cache files are ignored by the repository's default `.gitignore`.

## Use independently prepared embeddings

Save numeric arrays and Unicode strings with `numpy.savez`; the loader uses `allow_pickle=False`. Do not use object arrays.

| Key | Shape | Meaning |
| --- | --- | --- |
| `dataset` | Scalar string | Dataset identifier |
| `image_features` | `N x d` | One frozen embedding per distinct image |
| `labels` | `N` integers | Numeric labels in `[0, C)` |
| `text_features` | `C x d` | Row `c` is the initial text classifier for numeric label `c` |
| `class_names` | `C` unique strings | Names in the same class order as the text rows |
| `logit_scale` | Positive scalar | Multiplier applied to logits, already exponentiated |
| `test_idx` | One-dimensional integers | Shared test rows in the image bank |
| `train_k4_s1`, `val_k4_s1` | One-dimensional integers | Support and validation rows for 4-shot, seed 1 |
| `image_ids` | `N` unique strings, optional | Stable image identifiers or relative paths |
| `backbone`, `model_id`, `template`, `preprocess` | Scalar strings, optional for direct optimization | Feature provenance |

Include corresponding `train_kK_sS` and `val_kK_sS` keys for all requested shots and seeds. The formal grid requires `K in {1,4,16}` and `S in {1,2,3}`. Within each split, support, validation and test indices must be nonempty, unique and disjoint. Each class has exactly `K` support images and between 1 and `min(K,4)` validation images.

Direct `run_experiment.py --cache ...` accepts the schema above. To let `run_formal.py` recognize a bank for reuse, also provide the matching dataset name, `backbone="ViT-B/16"`, `template="a photo of a {}."` and all formal split keys.

Feature arrays must be finite and have nonzero row norms. A bank can be re-normalized on loading, but this does not replace the need to match the original encoder, preprocessing and class ordering. Class prototypes, logits or summary accuracies cannot substitute for the individual image embeddings.

## Original cache identifiers

The released configuration files retain SHA-256 hashes of the original feature banks. These hashes support provenance checks; the files themselves are not included or remotely fetched by the optimization code. Equivalent arrays serialized into a new NPZ file may have a different file hash.
