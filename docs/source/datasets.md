# Datasets

This project trains and evaluates SQL injection detectors on the
[**Superviz26-SQL**](https://zenodo.org/records/21068333) dataset
(DOI: `10.5281/zenodo.21068333`). Every build — the main LODO/in-domain scenarios, the
concept-drift CSVs, and the few-shot CSVs — ships inside a single `superviz26.zip`
archive on that record.

## Layout

Eight CSV files four for in-domain evaluation and four for leave-one-domain-out (LODO):

| File        | Type      | Train domains                           | Test domain    |
| ----------- | --------- | --------------------------------------- | -------------- |
| `a-a.csv`   | in-domain | OurAirports                             | OurAirports    |
| `b-b.csv`   | in-domain | Sakila                                  | Sakila         |
| `c-c.csv`   | in-domain | AdventureWorks                          | AdventureWorks |
| `d-d.csv`   | in-domain | OracleHR                                | OracleHR       |
| `bcd-a.csv` | LODO      | Sakila + AdventureWorks + OracleHR      | OurAirports    |
| `acd-b.csv` | LODO      | OurAirports + AdventureWorks + OracleHR | Sakila         |
| `abd-c.csv` | LODO      | OurAirports + Sakila + OracleHR         | AdventureWorks |
| `abc-d.csv` | LODO      | OurAirports + Sakila + AdventureWorks   | OracleHR       |

Each scenario ships as one CSV whose `split` column separates the benign training rows
(`split == "train"`) from the test rows (`split == "test"`, a mix of benign and attack
queries). The detectors train on the **full-split** build of these scenarios, with every
split drawn from the full per-domain pools (see [Acquisition](#acquisition) below).

## Acquisition

The default `superviz26` family is the **Superviz26-SQL-LODO** dataset (MLflow experiment
`Superviz26-SQL`). Its **full-split** CSVs are read from `~/datasets/superviz26-lodo/`.
They are either downloaded from the Zenodo archive (the `main` group) or generated locally
by the dataset generator's full-split mode — every scenario is drawn from the full
per-domain pools and downsampled to the smallest domain so all stay equal-sized:

```bash
# Download the 8 scenario CSVs from Zenodo:
python -m tools.fetch_superviz26 --groups main     # or: invoke fetch-superviz26

# ...or generate them locally, in the legacy-sqlia-dataset-generator repo:
python experiments/generate_splits.py --full
```

A missing file raises a `FileNotFoundError` pointing back at both options (the `main`
loader does not auto-download). The shared column/scenario metadata and the archive
checksums come from the vendored
[`MANIFEST.json`](https://github.com/gquetel/mlops-sqldetect/blob/main/data/raw/superviz26/MANIFEST.json).

## Loading

Use the typed loader rather than constructing paths by hand:

```python
from mlops_sqldetect.datasets import Superviz26, load_split

df_train = load_split(Superviz26.A_A, "train")
df_test  = load_split(Superviz26.A_A, "test")
```

By default only `full_query`, `label`, and `split` are read from disk.

When loading from the default (published) root, the loader verifies the CSV's size and
SHA-256/MD5 against `MANIFEST.json` before reading, so training only ever runs on bytes
matching the Zenodo archive; a mismatch (a modified or regenerated file) raises
`ValueError`. The check is memoised per file per process and skipped for custom roots
(caller-managed).

## Concept-drift protocol (`superviz26-drift`)

The same-domain concept-drift protocol re-partitions a *single* domain through its
per-template metadata to simulate an abrupt within-domain shift. For each domain the
query templates are split 50/50 per statement type into an **origin** set (S1) and a
held-out **shifted** set (S2). A detector is trained on the benign S1 train rows and
evaluated twice — on the S1 test set (reference) and on the never-seen S2 set
(post-drift); drift robustness is the AUROC drop `Δ = AUROC(S1) − AUROC(S2)`.

The four per-domain CSVs (`a.csv` … `d.csv`) carry the standard Superviz26 columns
plus a `drift_set` column (`origin`/`shifted`) that, together with `split`, tells the
three partitions apart:

| `(split, drift_set)` | Partition       | Use                                  |
| -------------------- | --------------- | ------------------------------------ |
| `(train, origin)`    | `origin_train`  | train the detector (benign rows)     |
| `(test,  origin)`    | `origin_test`   | S1 reference AUROC                    |
| `(test,  shifted)`   | `shifted_test`  | S2 post-drift AUROC                  |

The CSVs are read from `~/datasets/superviz26-cd/`. They are not vendored; either
build them locally with `experiments/build_concept_drift.py` (legacy generator) or
download the pre-built copies (see *Heavy supplementary datasets* below). Load them
with the three-way loader:

```python
from mlops_sqldetect.datasets.superviz26_drift import Superviz26Drift, load_drift

origin_train, origin_test, shifted_test = load_drift(Superviz26Drift.A)
```

Run the whole grid (15 methods × 4 domains) with the dedicated evaluator, which
trains once per cell and scores both test sets:

```bash
python -m mlops_sqldetect.evaluate_drift --methods ocsvm,lof,ae --extractors li,loginov,cv,sbert,codet5
```

Each cell appends one row — `auroc_s1`, `auroc_s2`, `delta_auroc`, … — to
`reports/superviz26-drift_results.csv`; average the per-domain rows to obtain the
per-method drift table. The grid also fans out to SLURM (see *Running on SLURM*).

## Heavy supplementary datasets (`drift`, `fsl`)

The concept-drift (`superviz26-drift`, the **Superviz26-SQL-CD** dataset) and few-shot
(`superviz26-fsl`) experiments read multi-GB CSVs that are **not** part of the default
`fetch_data` flow. The drift CSVs land in `~/datasets/superviz26-cd/`, the few-shot
in-domain CSVs in `~/datasets/superviz26-fsl/`.

They live in the same `superviz26.zip` archive on Zenodo
([record 21068333](https://zenodo.org/records/21068333), DOI `10.5281/zenodo.21068333`)
as the main dataset. The helper downloads that archive once (MD5-verified, resumable),
extracts the requested group(s) into their loader default roots, and verifies each CSV's
size and SHA-256 against
[`data/raw/superviz26/MANIFEST.json`](https://github.com/gquetel/mlops-sqldetect/blob/main/data/raw/superviz26/MANIFEST.json).

```bash
# Download the zip and extract the drift + few-shot CSVs
python -m tools.fetch_superviz26 --groups drift,fsl

# Verify already-extracted files without downloading
python -m tools.fetch_superviz26 --groups drift,fsl --check
```

The zip is deleted after a successful extraction; pass `--keep-archive` to retain it
(handy, since every group comes from the same 2.4 GB download). Equivalent invoke alias:

```bash
invoke fetch-supplementary           # both groups; or --groups drift
```

You usually do not need to run this by hand: the `superviz26-drift` and `superviz26-fsl`
loaders **auto-download** their group the first time a file is missing from the default
root, then reuse it on subsequent runs. Auto-fetch only triggers for the default root
(a custom `root=` is yours to manage). The default `superviz26` (LODO) loader does
**not** auto-download — its CSVs are fetched explicitly or generated locally (see
*Acquisition*).