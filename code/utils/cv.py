"""
Cross-validation folds fixed by 06_splits.py. Every model uses these, so fold-by-fold
comparisons between models are paired.

  fold_ids(df, scheme, seed)  -> array of fold numbers (0..K-1) aligned with df rows
  iter_folds(df, scheme, seed) -> yields (fold, train_index, val_index)
scheme = "municipality" (main; whole municipalities held out), "random" (comparison only) or
"district" (leave-one-district-out, D13: one fold per development district, fold number = position of
the district in sorted district IDs; no seed, the folds are fixed by the data).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

SCHEMES = ("municipality", "random")


def fold_ids(df, scheme="municipality", seed=C.SEED):
    col = f"fold_s{seed}"
    if scheme == "municipality":
        f = pd.read_csv(C.PROCESSED_DIR / "folds_municipality.csv").set_index(C.GROUP_COL)[col]
        out = df[C.GROUP_COL].map(f)
    elif scheme == "random":
        f = pd.read_parquet(C.PROCESSED_DIR / "folds_random.parquet").set_index("building_id")[col]
        out = df["building_id"].map(f)
    elif scheme == "district":
        assert C.FINAL_TEST_DISTRICT_ID not in set(df["district_id"])
        ids = sorted(df["district_id"].unique())
        out = df["district_id"].map({d: k for k, d in enumerate(ids)})
    else:
        raise ValueError(scheme)
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} rows have no fold (are these development rows?)")
    return out.to_numpy(dtype=int)


def iter_folds(df, scheme="municipality", seed=C.SEED):
    f = fold_ids(df, scheme, seed)
    for k in range(C.N_FOLDS):
        yield k, np.flatnonzero(f != k), np.flatnonzero(f == k)
