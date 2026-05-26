# Datasets

This project trains and evaluates SQL injection detectors on the
[**Superviz26-SQL**](https://zenodo.org/records/19627322) dataset
(DOI: `10.5281/zenodo.19627322`).

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

Each file holds ~1.1 M rows: 100 K benign training queries (`split == "train"`)
and 1 M test samples (`split == "test"`, ~90 % benign / 10 % attack).

## Acquisition

The CSVs are not vendored; download them with the helper script:

```bash
# All 8 files
python -m tools.fetch_superviz26

# Re-verify existing files
python -m tools.fetch_superviz26 --check
```

The script streams each file to `data/raw/superviz26/`, resumes interrupted downloads, and verifies sizes plus SHA-256 sums against [`MANIFEST.json`](https://github.com/gquetel/mlops-sqldetect/blob/main/data/raw/superviz26/MANIFEST.json). The manifest is the single source of truth — editing it changes what the script accepts.

Equivalent invoke alias:

```bash
invoke fetch-data --datasets a-a,bcd-a
```

## Loading

Use the typed loader rather than constructing paths by hand:

```python
from mlops_sqldetect.datasets import Superviz26, load_split

df_train = load_split(Superviz26.A_A, "train")
df_test  = load_split(Superviz26.A_A, "test")
```

By default only `full_query`, `label`, and `split` are read from disk. 