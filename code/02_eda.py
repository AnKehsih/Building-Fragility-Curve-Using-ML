"""
02_eda.py

Purpose : Exploratory data analysis on the 10 DEVELOPMENT districts only (the final-test
          district, Kavrepalanchok, is excluded by utils.data.load_development()).
          Describes the target, each allowed input and its relation to damage, data-quality
          oddities, and municipality sizes (for grouped-CV balance). Nothing is fitted.
Inputs  : data/processed/buildings_clean.parquet (via utils.data)
Outputs : results/tables/02_grade_prevalence_dev.csv
          results/tables/02_numeric_summary.csv
          results/tables/02_numeric_spearman.csv
          results/tables/02_quality_flags.csv
          results/tables/02_categorical_levels.csv
          results/tables/02_superstructure.csv
          results/tables/02_storeys_age.csv
          results/tables/02_municipality_summary.csv
          results/figures/02_grades_by_district.png
          results/figures/02_severe_vs_storeys_age.png
Run     : .venv/Scripts/python code/02_eda.py     (from the project root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development
from utils import plotting as P

T, FIG = C.TABLES_DIR, C.FIGURES_DIR
df = load_development()
df["severe"] = (df[C.TARGET] >= C.SEVERE_MIN_GRADE).astype(int)
df["district"] = df["district_id"].map(C.DISTRICTS)
superstructure = [c for c in df.columns if c.startswith(C.SUPERSTRUCTURE_PREFIX)]
print(f"Development data: {len(df):,} buildings, {df['district_id'].nunique()} districts, "
      f"{df['vdcmun_id'].nunique()} municipalities")


def by_level(series):
    """n, share of buildings, mean grade and share severe for each level of a column."""
    g = df.groupby(series, dropna=False)
    out = pd.DataFrame({"n": g.size(), "mean_grade": g[C.TARGET].mean(), "share_severe": g["severe"].mean()})
    out["share_of_buildings"] = out["n"] / len(df)
    return out.round(3)


# ------------------------------------------------------------------ target prevalence
prev = df[C.TARGET].value_counts().sort_index().rename("n").to_frame()
prev["share"] = (prev["n"] / len(df)).round(4)
prev.to_csv(T / "02_grade_prevalence_dev.csv")
print("\nGrade prevalence (development):\n", prev.to_string())
print(f"Share severe (grade >= {C.SEVERE_MIN_GRADE}): {df['severe'].mean():.3f}")

# ------------------------------------------------------------------ numeric inputs
q = [0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1]
num = df[C.NUMERIC_FEATURES].quantile(q).T
num.columns = [f"q{x:g}" for x in q]
num.insert(0, "missing", df[C.NUMERIC_FEATURES].isna().sum())
num.insert(1, "mean", df[C.NUMERIC_FEATURES].mean())
num.round(2).to_csv(T / "02_numeric_summary.csv")
print("\nNumeric inputs:\n", num.round(1).to_string())

spear = df[C.NUMERIC_FEATURES + [C.TARGET]].corr(method="spearman").round(3)
spear.to_csv(T / "02_numeric_spearman.csv")
print("\nSpearman correlations:\n", spear.to_string())

# ------------------------------------------------------------------ data-quality flags (descriptive; nothing removed)
ft_per_storey = df["height_ft_pre_eq"] / df["count_floors_pre_eq"]
flags = {
    "height_ft_pre_eq == 0": int((df["height_ft_pre_eq"] == 0).sum()),
    "height per storey < 5 ft": int((ft_per_storey < 5).sum()),
    "height per storey > 20 ft": int((ft_per_storey > 20).sum()),
    "count_floors_pre_eq == 0": int((df["count_floors_pre_eq"] == 0).sum()),
    "count_floors_pre_eq > 5": int((df["count_floors_pre_eq"] > 5).sum()),
    "plinth_area_sq_ft < 100": int((df["plinth_area_sq_ft"] < 100).sum()),
    "plinth_area_sq_ft > 3000": int((df["plinth_area_sq_ft"] > 3000).sum()),
    "age_building missing (>=995 code)": int(df["age_building"].isna().sum()),
    "no superstructure flag set": int((df[superstructure].sum(axis=1) == 0).sum()),
    "more than one superstructure flag set": int((df[superstructure].sum(axis=1) > 1).sum()),
}
flags = pd.Series(flags, name="n_buildings").to_frame()
flags["share"] = (flags["n_buildings"] / len(df)).round(4)
flags.to_csv(T / "02_quality_flags.csv")
print("\nData-quality flags:\n", flags.to_string())

# ------------------------------------------------------------------ categorical inputs
cat_rows = []
for col in C.CATEGORICAL_FEATURES:
    t = by_level(df[col]).reset_index().rename(columns={col: "level"})
    t.insert(0, "feature", col)
    cat_rows.append(t)
cats = pd.concat(cat_rows, ignore_index=True)
cats.to_csv(T / "02_categorical_levels.csv", index=False)
print("\nCategorical inputs (levels, n, mean grade, share severe):\n", cats.to_string(index=False))

# ------------------------------------------------------------------ superstructure flags
ss = pd.DataFrame({
    "n_with_flag": df[superstructure].sum(),
    "share_with_flag": df[superstructure].mean(),
    "share_severe_if_flag": [df.loc[df[c] == 1, "severe"].mean() for c in superstructure],
    "share_severe_if_not": [df.loc[df[c] == 0, "severe"].mean() for c in superstructure],
}).round(3)
ss.index = [c.replace(C.SUPERSTRUCTURE_PREFIX, "") for c in ss.index]
ss.to_csv(T / "02_superstructure.csv")
print("\nSuperstructure flags:\n", ss.to_string())

# ------------------------------------------------------------------ storeys and age bands
storey_band = df["count_floors_pre_eq"].clip(upper=5).astype(str).replace({"5": "5+"})
age_band = pd.cut(df["age_building"], [-1, 4, 9, 19, 29, 49, 994],
                  labels=["0-4", "5-9", "10-19", "20-29", "30-49", "50+"]).astype(object).fillna("missing")
sa = pd.concat([by_level(storey_band).assign(variable="storeys"),
                by_level(age_band).assign(variable="age_band")]).rename_axis("level").reset_index()
sa.to_csv(T / "02_storeys_age.csv", index=False)
print("\nStoreys and age bands:\n", sa.to_string(index=False))

# ------------------------------------------------------------------ municipality sizes (grouped-CV balance)
mg = df.groupby("vdcmun_id")
muni = pd.DataFrame({"district": mg["district"].first(), "n": mg.size(),
                     "share_severe": mg["severe"].mean(), "share_grade5": mg[C.TARGET].apply(lambda s: (s == 5).mean())})
muni.round(3).to_csv(T / "02_municipality_summary.csv")
print(f"\nMunicipalities: n buildings min {muni['n'].min():,}, median {int(muni['n'].median()):,}, "
      f"max {muni['n'].max():,}; share severe ranges {muni['share_severe'].min():.2f}-{muni['share_severe'].max():.2f}")
within = muni.groupby("district")["share_severe"].agg(["min", "max", "std"]).round(2)
print("Within-district spread of municipality severe share:\n", within.to_string())

# ------------------------------------------------------------------ figures
P.set_style()

# Fig 1: grade mix per development district (stacked, ordered by Grade 5 share)
shares = pd.crosstab(df["district"], df[C.TARGET], normalize="index")
shares = shares.loc[shares[5].sort_values().index]
n_by = df["district"].value_counts()
fig, ax = plt.subplots(figsize=(P.COL_WIDTH_IN, 2.4))
left = np.zeros(len(shares))
for k, color in zip(C.GRADES, P.GRADE_COLORS):
    ax.barh(range(len(shares)), shares[k], left=left, color=color, height=0.7,
            edgecolor=P.SURFACE, linewidth=1, label=f"Grade {k}")
    left += shares[k].values
ax.set_yticks(range(len(shares)), [f"{d} ({n_by[d]/1000:.0f}k)" for d in shares.index])
ax.set_xlim(0, 1)
ax.set_xlabel("Share of buildings")
ax.grid(axis="y", visible=False)
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.4, 1.0), handlelength=1, columnspacing=0.8)
P.save(fig, FIG / "02_grades_by_district.png")

# Fig 2: share severe vs storeys and vs age band (one series per panel)
fig, axes = plt.subplots(1, 2, figsize=(P.COL_WIDTH_IN, 1.7), sharey=True)
for ax, var, order, xlabel in [
        (axes[0], "storeys", ["1", "2", "3", "4", "5+"], "Storeys before earthquake"),
        (axes[1], "age_band", ["0-4", "5-9", "10-19", "20-29", "30-49", "50+"], "Age (years)")]:
    s = sa[sa["variable"] == var].set_index("level").loc[order]
    ax.plot(range(len(order)), s["share_severe"], marker="o", color=P.CATEGORICAL[0])
    ax.set_xticks(range(len(order)), order, rotation=45 if var == "age_band" else 0)
    ax.set_xlabel(xlabel)
axes[0].set_ylabel(f"Share grade $\\geq${C.SEVERE_MIN_GRADE}")
axes[0].set_ylim(0, 1)
P.save(fig, FIG / "02_severe_vs_storeys_age.png")
print("\nFigures written to results/figures/02_*.png")
