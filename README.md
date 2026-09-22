# SQL attack detection benchmarks

This project evaluates SQL attack detectors across feature extractors and Superviz datasets.

## Run locally

From the repository root, Nix can pin Python, uv, and system libraries. `uv sync` installs the Python packages
from `uv.lock`; activate that environment before running project commands. Nix is optional: without it, install
uv yourself, skip `nix-shell`, and activate `.venv/bin/activate` instead.

```sh
nix-shell
uv sync --frozen --extra cpu
source .venv-nix-cpu/bin/activate
invoke fetch-data
python -m sqlad_benchmarking.evaluate_suite \
  --dataset superviz25 --suite all --methods ocsvm --extractors li --limit 50000 --no-track
```

`invoke fetch-data` downloads Superviz25 to `data/raw/superviz25/dataset.csv` and the main Superviz26 CSVs to
`~/datasets/superviz26-lodo/`. The evaluation writes `reports/superviz25_results.csv` and a log under
`reports/superviz25/logs/`. Omit `--limit` to use the full CSV.

See [Datasets](docs/source/datasets.md) for the other dataset families and
[SLURM](docs/source/slurm.md) for parallel runs. CLI options are available with
`python -m sqlad_benchmarking.evaluate_suite --help` in the activated environment.

## Environments

For a GPU, enter `SQLAD_EXTRA=cu126 nix-shell`, run `uv sync --frozen --extra cu126`, and source
`.venv-nix-cu126/bin/activate`. Use `cu130` for Blackwell GPUs. Nix keeps each profile in its own
`.venv-nix-<profile>` directory. Without Nix, uv uses `.venv`; syncing another extra replaces the packages
in that environment, so keep separate environments if you need several builds at once.
On a SLURM cluster, source `tools/setup-env.sh` to build its separate environments as described in the
[SLURM guide](docs/source/slurm.md). Always sync with the matching uv extra so it keeps the intended PyTorch build.

## Credits

The repository structure is based on [mlops_template](https://github.com/SkafteNicki/mlops_template).
