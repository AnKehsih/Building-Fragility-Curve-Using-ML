"""
06_splits.py

Purpose : Fix the cross-validation folds for all development modelling (CONTEXT 5.2).
          Main scheme  : 5 folds of whole MUNICIPALITIES, 3 seeds. Per seed: N_DRAWS random orders of
                         the municipalities; each order is dealt greedily to the fold with the fewest
                         buildings so far (size balance); the draw with the best grade balance is kept.
                         (sklearn StratifiedGroupKFold was tried first and abandoned: its group order
                         is deterministic, so all three seeds gave identical folds, ARI = 1.0.)
          Comparison   : ordinary random 5-fold split of buildings (KFold), same 3 seeds, used only
                         to show how much a random split overstates performance.
          Folds are fixed once here and read by every later script, so all models see identical splits
          (paired fold-by-fold comparisons).
Inputs  : data/processed/buildings_clean.parquet (via utils.data: development districts only)
Outputs : data/processed/folds_municipality.csv   vdcmun_id, fold_s<seed> (0..4)
          data/processed/folds_random.parquet      building_id, fold_s<seed>
          results/tables/06_fold_balance.csv        n, municipalities, districts, grade shares per fold
Run     : .venv/Scripts/python code/06_splits.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development

N_DRAWS = 2000

df = load_development()
y, groups = df[C.TARGET].values, df[C.GROUP_COL].values
# grade counts per municipality (rows) -> used for size and grade balance
counts = pd.crosstab(df[C.GROUP_COL], df[C.TARGET])
overall = counts.sum() / counts.values.sum()


def balanced_group_folds(rng):
    """Best of N_DRAWS random greedy assignments of municipalities to folds.
    Cost = largest absolute deviation of any fold's grade share from the overall share."""
    best, best_cost = None, np.inf
    ids, cnt = counts.index.values, counts.values
    for _ in range(N_DRAWS):
        order = rng.permutation(len(ids))
        fold_n, fold_cnt = np.zeros(C.N_FOLDS), np.zeros((C.N_FOLDS, cnt.shape[1]))
        assign = np.empty(len(ids), dtype=int)
        for i in order:
            k = int(np.argmin(fold_n))
            assign[i], fold_n[k] = k, fold_n[k] + cnt[i].sum()
            fold_cnt[k] += cnt[i]
        cost = np.abs(fold_cnt / fold_n[:, None] - overall.values).max()
        if cost < best_cost:
            best, best_cost = assign.copy(), cost
    return pd.Series(best, index=ids), best_cost


muni_folds = pd.DataFrame(index=counts.index)
rand_folds = pd.DataFrame({"building_id": df["building_id"].values})
balance = []
for seed in C.CV_REPEAT_SEEDS:
    assign, cost = balanced_group_folds(np.random.default_rng(seed))
    print(f"seed {seed}: largest grade-share deviation of any fold = {cost:.4f}")
    fold = df[C.GROUP_COL].map(assign).values
    # every municipality must sit in exactly one fold
    per_muni = pd.Series(fold).groupby(groups).nunique()
    assert (per_muni == 1).all()
    muni_folds[f"fold_s{seed}"] = pd.Series(fold).groupby(groups).first()

    rfold = np.empty(len(df), dtype=int)
    for k, (_, val) in enumerate(KFold(C.N_FOLDS, shuffle=True, random_state=seed).split(df)):
        rfold[val] = k
    rand_folds[f"fold_s{seed}"] = rfold

    for scheme, f in [("municipality", fold), ("random", rfold)]:
        for k in range(C.N_FOLDS):
            sub = df[f == k]
            row = {"scheme": scheme, "seed": seed, "fold": k, "n": len(sub),
                   "n_municipalities": sub[C.GROUP_COL].nunique(), "n_districts": sub["district_id"].nunique(),
                   "share_severe": (sub[C.TARGET] >= C.SEVERE_MIN_GRADE).mean()}
            row.update({f"share_g{g}": (sub[C.TARGET] == g).mean() for g in C.GRADES})
            balance.append(row)

muni_folds.reset_index().to_csv(C.PROCESSED_DIR / "folds_municipality.csv", index=False)
rand_folds.to_parquet(C.PROCESSED_DIR / "folds_random.parquet", index=False)
balance = pd.DataFrame(balance).round(4)
balance.to_csv(C.TABLES_DIR / "06_fold_balance.csv", index=False)

print(f"Development: {len(df):,} buildings, {len(muni_folds)} municipalities, {C.N_FOLDS} folds x seeds {C.CV_REPEAT_SEEDS}")
print(balance[balance["scheme"] == "municipality"].to_string(index=False))
summ = balance.groupby("scheme")[["n", "share_severe", "share_g1", "share_g5"]].agg(["min", "max"]).round(3)
print("\nRange across folds (all seeds):\n", summ.to_string())
# Seeds must give genuinely different fold assignments (adjusted Rand index well below 1).
s = C.CV_REPEAT_SEEDS
ari = {f"{a}-{b}": round(adjusted_rand_score(muni_folds[f'fold_s{a}'], muni_folds[f'fold_s{b}']), 3)
       for i, a in enumerate(s) for b in s[i + 1:]}
print("\nAdjusted Rand index between seeds' municipality assignments:", ari)
assert max(ari.values()) < 0.9, "seeds produced (nearly) identical folds"
