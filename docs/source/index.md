# SQL attack detection benchmarks

The project trains and evaluates SQL attack detectors on the Superviz datasets. From the repository root,
Nix can pin Python, uv, and system libraries. Without Nix, install uv yourself, skip `nix-shell`, and activate
`.venv/bin/activate` instead. Build and activate the locked CPU environment before running commands:

```sh
nix-shell
uv sync --frozen --extra cpu
source .venv-nix-cpu/bin/activate
invoke fetch-data
python -m sqlad_benchmarking.evaluate_suite \
  --dataset superviz25 --suite all --methods ocsvm --extractors li --limit 50000 --no-track
```

`invoke fetch-data` saves Superviz25 to `data/raw/superviz25/dataset.csv`
and the main Superviz26 CSVs to `~/datasets/superviz26-lodo/`. Evaluation writes
`reports/superviz25_results.csv` and a log in `reports/superviz25/logs/`.
Remove `--limit` to evaluate every row.

See [Datasets](datasets.md) for acquisition and scenarios, or [SLURM](slurm.md) for cluster runs and MLflow results.
