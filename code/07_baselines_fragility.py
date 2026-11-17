"""
07_baselines_fragility.py

Purpose : Cross-validated baselines on the 10 development districts (CONTEXT 4, 5.2, 5.3):
            majority  - training-fold grade distribution for every building
            fragility - lognormal fragility curves per structural type (utils/fragility.py),
                        fitted on training folds only, with each PGA variant:
                        {preferred, neic_v9} ShakeMap x {municipality, district} averaging.
          Schemes: municipality-grouped folds (main) and random folds (comparison), 3 seeds each.
          Also fits the curves once on ALL development data (for plots / literature comparison;
          not used for any score), and saves out-of-fold predictions (seed 678, municipality folds)
          for later calibration plots and paired comparisons with the ML models.
Inputs  : data/processed/buildings_clean.parquet, pga_municipality.csv, pga_district.csv,
          folds_municipality.csv, folds_random.parquet
Outputs : results/tables/07_cv_metrics.csv          one row per model x PGA variant x scheme x seed x fold
          results/tables/07_cv_summary.csv          mean and SD over folds
          results/tables/07_paired_differences.csv  fold-by-fold RPS differences vs primary fragility
          results/tables/07_fragility_params_cv.csv per fold fitted beta, theta
          results/tables/07_fragility_params_full_dev.csv
          data/processed/oof/<model>_s678.parquet   building_id, fold, p1..p5
          results/figures/07_fragility_curves.png
Run     : .venv/Scripts/python code/07_baselines_fragility.py > results/tables/07_console_output.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development, attach_pga
from utils.cv import iter_folds, SCHEMES
from utils.fragility import MajorityModel, FragilityModel, grade_probs
from utils.metrics import evaluate
from utils.preprocessing import assign_structural_type
from utils import plotting as PL

T, OOF = C.TABLES_DIR, C.PROCESSED_DIR / "oof"
OOF.mkdir(exist_ok=True)
PRIMARY = f"fragility_{C.SHAKEMAP_PRIMARY}_municipality"

base = load_development()
base["structural_type"] = assign_structural_type(base, C.STRUCTURAL_TYPE_RULE)
y = base[C.TARGET].to_numpy()
variants = {f"{src}_{lvl}": attach_pga(base, src, lvl)["pga"].to_numpy()
            for src in (C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY) for lvl in ("municipality", "district")}

rows, param_rows = [], []
for scheme in SCHEMES:
    for seed in C.CV_REPEAT_SEEDS:
        oof = {}
        for k, tr, va in iter_folds(base, scheme, seed):
            P = MajorityModel().fit(y[tr]).predict_proba(len(va))
            rows.append({"model": "majority", "scheme": scheme, "seed": seed, "fold": k, **evaluate(P, y[va])})
            oof.setdefault("majority", []).append((va, k, P))
            for vname, pga in variants.items():
                d = base[["structural_type", C.TARGET]].assign(pga=pga)
                fm = FragilityModel().fit(d.iloc[tr])
                P = fm.predict_proba(d.iloc[va])
                name = f"fragility_{vname}"
                rows.append({"model": name, "scheme": scheme, "seed": seed, "fold": k, **evaluate(P, y[va])})
                oof.setdefault(name, []).append((va, k, P))
                param_rows.append(fm.table().assign(model=name, scheme=scheme, seed=seed, fold=k))
        if scheme == "municipality" and seed == C.SEED:          # save out-of-fold predictions
            for name, parts in oof.items():
                idx = np.concatenate([p[0] for p in parts])
                out = pd.DataFrame(np.vstack([p[2] for p in parts]).astype("float32"),
                                   columns=[f"p{g}" for g in C.GRADES])
                out.insert(0, "fold", np.concatenate([np.full(len(p[0]), p[1]) for p in parts]))
                out.insert(0, "building_id", base["building_id"].to_numpy()[idx])
                out.to_parquet(OOF / f"{name}_s{seed}.parquet", index=False)
        print(f"done: {scheme} seed {seed}")

res = pd.DataFrame(rows)
res.to_csv(T / "07_cv_metrics.csv", index=False)
pd.concat(param_rows).round(4).to_csv(T / "07_fragility_params_cv.csv", index=False)

metrics = ["rps", "brier_severe", "log_loss", "qwk", "macro_f1", "accuracy"]
summ = res.groupby(["scheme", "model"])[metrics].agg(["mean", "std"]).round(4)
summ.to_csv(T / "07_cv_summary.csv")
print("\nCV summary over 15 folds (3 seeds x 5), mean and SD:")
for scheme in SCHEMES:
    print(f"\n[{scheme} folds]\n", summ.loc[scheme].sort_values(("rps", "mean")).to_string())

# fold-by-fold RPS differences relative to the primary fragility model (negative = better than primary)
key = ["scheme", "seed", "fold"]
ref = res[res["model"] == PRIMARY].set_index(key)["rps"]
diff = res.set_index(key).assign(rps_minus_primary=lambda d: d["rps"] - ref.reindex(d.index).values).reset_index()
pdiff = diff.groupby(["scheme", "model"])["rps_minus_primary"].agg(
    mean="mean", sd="std", min="min", max="max",
    n_folds_better=lambda s: int((s < 0).sum())).round(4)
pdiff.to_csv(T / "07_paired_differences.csv")
print(f"\nFold-by-fold RPS minus {PRIMARY} (15 paired folds per scheme):\n", pdiff.to_string())

# random vs municipality folds: how much does a random split flatter each model?
gap = (summ.loc["random"][("rps", "mean")] - summ.loc["municipality"][("rps", "mean")]).round(4)
print("\nRPS(random folds) - RPS(municipality folds), negative = random split looks better:\n", gap.to_string())

# ------------------------------------------------------------------ full-development fit (plots only)
full = []
for src in (C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY):
    d = base[["structural_type", C.TARGET]].assign(pga=variants[f"{src}_municipality"])
    full.append(FragilityModel().fit(d).table().assign(pga_source=src))
full = pd.concat(full).round(4)
full.to_csv(T / "07_fragility_params_full_dev.csv", index=False)
print("\nFragility parameters fitted on all development data (theta in g):\n", full.to_string(index=False))

PL.set_style()
pt = full[full["pga_source"] == C.SHAKEMAP_PRIMARY].set_index("structural_type")
show = ["stone_masonry", "adobe_mud", "mud_mortar_brick", "mixed_strong_weak",
        "cement_mortar_masonry", "rc_non_engineered", "rc_engineered"]
x = np.linspace(0.05, 1.2, 200)
fig, ax = plt.subplots(figsize=(PL.COL_WIDTH_IN, 2.2))
for c, color in zip(show, PL.CATEGORICAL):
    p = pt.loc[c]
    ln_theta = np.log(p[["theta_g2", "theta_g3", "theta_g4", "theta_g5"]].to_numpy(float))
    sev = grade_probs(np.log(x), p["beta"], ln_theta)[:, 3:].sum(1)
    ax.plot(x, sev, color=color, label=c.replace("_", " "))
pga_m = pd.read_csv(C.PROCESSED_DIR / "pga_municipality.csv")[f"pga_{C.SHAKEMAP_PRIMARY}"]
ax.axvspan(pga_m.min(), pga_m.max(), color=PL.GRID, alpha=0.5, lw=0, label="observed PGA range")
ax.set_xlabel(f"PGA (g), {C.SHAKEMAP_PRIMARY} ShakeMap")
ax.set_ylabel("P(grade $\\geq$ 4)")
ax.set_ylim(0, 1)
ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
PL.save(fig, C.FIGURES_DIR / "07_fragility_curves.png")
