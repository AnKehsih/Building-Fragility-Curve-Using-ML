"""
03_structural_type.py

Purpose : Decision D4. Check how often the has_superstructure_* flags overlap, then compare
          three candidate rules for assigning ONE structural type per building (defined in
          utils/preprocessing.py): "precedence" (CONTEXT 5.1 preliminary rule),
          "precedence_mixed" (adds a strong+weak mixed class) and "weakest_first".
          A rule is adequate if buildings that mix load-bearing materials behave like the
          "pure" buildings of the type they are assigned to.
          DEVELOPMENT districts only. Nothing is fitted; labels are used descriptively.
Inputs  : data/processed/buildings_clean.parquet (via utils.data)
Outputs : results/tables/03_flag_count_distribution.csv
          results/tables/03_flag_combinations_top20.csv
          results/tables/03_rule_comparison.csv           one row per candidate rule
          results/tables/03_types_<rule>.csv              n, grade shares, pure vs mixed, per type
          results/figures/03_grades_by_structural_type.png   (rule in config.STRUCTURAL_TYPE_RULE)
Run     : .venv/Scripts/python code/03_structural_type.py     (from the project root)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development
from utils.preprocessing import (assign_structural_type, n_load_bearing_materials,
                                 type_order, RULES, P)
from utils import plotting as PL

T, FIG = C.TABLES_DIR, C.FIGURES_DIR
df = load_development()
ss = [c for c in df.columns if c.startswith(P)]
names = [c.replace(P, "") for c in ss]
y = df[C.TARGET]
df["severe"] = (y >= C.SEVERE_MIN_GRADE).astype(int)

# ------------------------------------------------------------------ 1. how much do flags overlap?
nflags = df[ss].sum(axis=1)
dist = nflags.value_counts().sort_index().rename("n").to_frame()
dist["share"] = (dist["n"] / len(df)).round(4)
dist.to_csv(T / "03_flag_count_distribution.csv")
print(f"Buildings with >1 flag: {int((nflags > 1).sum()):,} ({(nflags > 1).mean():.1%})")

combo = df[ss].astype(str).agg("".join, axis=1)
counts = combo.value_counts().head(20)
top = pd.DataFrame({"combination": ["+".join(n for n, b in zip(names, k) if b == "1") for k in counts.index],
                    "n": counts.values, "share": (counts.values / len(df)).round(4),
                    "share_severe": [round(df.loc[combo == k, "severe"].mean(), 3) for k in counts.index]})
top.to_csv(T / "03_flag_combinations_top20.csv", index=False)
print("\nTop flag combinations:\n", top.to_string(index=False))

nlb = n_load_bearing_materials(df)
df["mixed"] = np.where(nlb > 1, "mixed", "pure")
print(f"\nBuildings with >1 LOAD-BEARING material flag (ignoring timber/bamboo/other): "
      f"{int((nlb > 1).sum()):,} ({(nlb > 1).mean():.1%})")


# ------------------------------------------------------------------ 2. compare candidate rules
def eta_squared(groups, values):
    """Share of the variance of `values` explained by group means (between / total)."""
    m = values.mean()
    g = values.groupby(groups, observed=True)
    return float((g.size() * (g.mean() - m) ** 2).sum() / ((values - m) ** 2).sum())


summary, tables = [], {}
for rule in RULES:
    st = pd.Series(assign_structural_type(df, rule), index=df.index, name="structural_type")
    order = type_order(rule)
    t = pd.crosstab(st, y, normalize="index").round(3)
    t.columns = [f"share_grade{k}" for k in t.columns]
    t.insert(0, "n", st.value_counts())
    t["share_severe"] = df.groupby(st, observed=True)["severe"].mean().round(3)
    pm = df.groupby([st, "mixed"], observed=True).agg(n=("severe", "size"), sev=("severe", "mean")).unstack("mixed")
    t["n_mixed"] = pm[("n", "mixed")] if ("n", "mixed") in pm else np.nan
    t["share_severe_pure"] = pm[("sev", "pure")].round(3)
    t["share_severe_mixed"] = pm[("sev", "mixed")].round(3) if ("sev", "mixed") in pm else np.nan
    t["gap_mixed_minus_pure"] = (t["share_severe_mixed"] - t["share_severe_pure"]).round(3)
    t = t.reindex(order)
    t.to_csv(T / f"03_types_{rule}.csv")
    tables[rule] = t

    both = t.dropna(subset=["gap_mixed_minus_pure"])
    summary.append({
        "rule": rule, "n_types": int((t["n"] > 0).sum()), "smallest_type_n": int(t["n"].min()),
        "eta2_grade": round(eta_squared(st, y.astype(float)), 4),
        "eta2_severe": round(eta_squared(st, df["severe"].astype(float)), 4),
        "mixed_weighted_abs_gap": round(float(np.average(both["gap_mixed_minus_pure"].abs(), weights=both["n_mixed"])), 3),
        "max_abs_gap": round(float(both["gap_mixed_minus_pure"].abs().max()), 3),
    })
    print(f"\n=== rule: {rule} ===\n", t.to_string())

summary = pd.DataFrame(summary).set_index("rule")
summary.to_csv(T / "03_rule_comparison.csv")
print("\nRule comparison (eta2 = share of variance explained by type; gap = severe share of "
      "mixed minus pure buildings within a type, weighted by n mixed):\n", summary.to_string())

# ------------------------------------------------------------------ figure: grade mix per type (chosen rule)
chosen = tables[C.STRUCTURAL_TYPE_RULE]
PL.set_style()
order = chosen.sort_values("share_severe").index
fig, ax = plt.subplots(figsize=(PL.COL_WIDTH_IN, 2.4))
left = np.zeros(len(order))
for k, color in zip(C.GRADES, PL.GRADE_COLORS):
    vals = chosen.loc[order, f"share_grade{k}"].values
    ax.barh(range(len(order)), vals, left=left, color=color, height=0.7,
            edgecolor=PL.SURFACE, linewidth=1, label=f"Grade {k}")
    left += vals
ax.set_yticks(range(len(order)), [f"{t.replace('_', ' ')} ({chosen.loc[t, 'n']/1000:.1f}k)" for t in order])
ax.set_xlim(0, 1)
ax.set_xlabel("Share of buildings")
ax.grid(axis="y", visible=False)
ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.3, 1.0), handlelength=1, columnspacing=0.8)
PL.save(fig, FIG / "03_grades_by_structural_type.png")
print(f"\nFigure (rule = {C.STRUCTURAL_TYPE_RULE}) written to results/figures/03_grades_by_structural_type.png")
