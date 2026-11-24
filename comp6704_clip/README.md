# Optimization implementation

See the [repository README](../README.md) for installation, commands, the experimental protocol and reported results. Commands in that README are run from the repository root.

See [feature preparation and cache format](../docs/feature_cache.md) for the optional real-image extraction path and the `.npz` interface.

The standalone optimizer has a default iteration limit of 3,000. Pass `--max-iter 10000` to reproduce the formal setting. `run_formal.py` supplies the formal settings automatically and is Linux-specific.
