# Datasets

The registered dataset families use CSV files with a `split` column. The `train` rows contain benign queries;
the `test` rows contain benign and attack queries.

| Family | Files | Default location |
| --- | --- | --- |
| `superviz25` | One scenario, `dataset.csv` | `data/raw/superviz25/` |
| `superviz26` | Four in-domain and four leave-one-domain-out (LODO) scenarios | `~/datasets/superviz26-lodo/` |
| `superviz26-big` | Eight scenarios with larger training sets | `~/datasets/superviz26-big/` |
| `superviz26-drift` | Four same-domain drift scenarios | `~/datasets/superviz26-cd/` |
| `superviz26-fsl` | Four target-domain few-shot scenarios | `~/datasets/superviz26-fsl/` |

From the repository root, build and activate the CPU environment, then fetch the standard datasets. Nix is
optional; without it, install uv yourself, skip `nix-shell`, and source `.venv/bin/activate` instead.

```sh
nix-shell
uv sync --frozen --extra cpu
source .venv-nix-cpu/bin/activate
invoke fetch-data
```

The drift and few-shot CSVs are several GB. Fetch them when needed:

```sh
invoke fetch-supplementary
```

`superviz25` comes from [Superviz25-SQL](https://zenodo.org/records/17086037). The `superviz26` main, drift,
and few-shot groups come from one [Superviz26-SQL archive](https://zenodo.org/records/21068333). The fetchers
verify the downloaded files against the manifests in `data/raw/`. `invoke fetch-data` fetches Superviz25 and
Superviz26 main; `invoke fetch-supplementary` fetches drift and few-shot data. The drift and few-shot loaders
also fetch their group when a file is missing from its default location.

For Superviz26, `a-a.csv` through `d-d.csv` train and test on the same domain. The LODO files
`bcd-a.csv`, `acd-b.csv`, `abd-c.csv`, and `abc-d.csv` train on three domains and test on the fourth.
The letters identify OurAirports (`a`), Sakila (`b`), AdventureWorks (`c`), and OracleHR (`d`).

## Generate Superviz26 Big

First fetch the Superviz26 main group. From the root of the
[dataset generator](https://github.com/gquetel/cross-domain-SQLAD-datasets-generation), run:

```sh
python experiments/alternative_datasets_builders/build_big_trainsets.py
```

The generator needs its `config.toml` and MySQL setup. It creates extra benign queries and writes the eight CSVs to
`~/datasets/superviz26-big/`. This family has no download command. Its evaluator uses the same
`--dataset superviz26-big` interface as the standard suite.
