"""
05b_pga_damage_check.py

Purpose : Descriptive check (nothing fitted) of how each ShakeMap's municipal PGA lines up with
          observed damage, on the 97 DEVELOPMENT municipalities. To remove the effect of building
          mix, damage is measured within stone masonry only (the dominant type, about 80%).
          This informs, but does not decide, the PGA source: the decision is made on
          development-CV scores of the fragility baseline (07) with each map.
Inputs  : data/processed/buildings_clean.parquet (via utils.data), data/processed/pga_municipality.csv
Outputs : results/tables/05b_pga_vs_damage.csv            per municipality: PGA (both maps), n, severe share
          results/tables/05b_pga_damage_correlation.csv   Spearman per map
          results/figures/05b_pga_vs_damage.png
Run     : .venv/Scripts/python code/05b_pga_damage_check.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development
from utils.preprocessing import assign_structural_type
from utils import plotting as PL

df = load_development()
df["structural_type"] = assign_structural_type(df, C.STRUCTURAL_TYPE_RULE)
stone = df[df["structural_type"] == "stone_masonry"]
pga = pd.read_csv(C.PROCESSED_DIR / "pga_municipality.csv")
maps = [C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY]

g = stone.groupby("vdcmun_id")
muni = pd.DataFrame({"n_stone": g.size(),
                     "share_severe_stone": g[C.TARGET].apply(lambda s: (s >= C.SEVERE_MIN_GRADE).mean()),
                     "share_grade5_stone": g[C.TARGET].apply(lambda s: (s == 5).mean())})
muni = muni.join(pga.set_index("vdcmun_id")[["district_name"] + [f"pga_{m}" for m in maps]])
muni.round(4).to_csv(C.TABLES_DIR / "05b_pga_vs_damage.csv")

rows = []
for m in maps:
    for target in ["share_severe_stone", "share_grade5_stone"]:
        rows.append({"map": m, "damage_measure": target,
                     "spearman_municipal": round(muni[f"pga_{m}"].corr(muni[target], method="spearman"), 3)})
corr = pd.DataFrame(rows)
corr.to_csv(C.TABLES_DIR / "05b_pga_damage_correlation.csv", index=False)
print(f"{len(muni)} development municipalities; stone-masonry buildings per municipality: "
      f"min {muni['n_stone'].min()}, median {int(muni['n_stone'].median())}")
print(corr.to_string(index=False))
by_d = muni.groupby("district_name")[[f"pga_{m}" for m in maps] + ["share_severe_stone"]].mean().round(3)
print("\nDistrict averages of municipal values (stone masonry):\n",
      by_d.sort_values("share_severe_stone").to_string())

PL.set_style()
fig, axes = plt.subplots(1, 2, figsize=(PL.COL_WIDTH_IN, 1.8), sharey=True)
for ax, m, title in zip(axes, maps, ["Preferred 2020 (primary)", "NEIC v9 2015"]):
    ax.scatter(muni[f"pga_{m}"], muni["share_severe_stone"], s=8, color=PL.CATEGORICAL[0],
               edgecolor=PL.SURFACE, linewidth=0.5)
    rho = corr.query("map == @m and damage_measure == 'share_severe_stone'")["spearman_municipal"].iat[0]
    ax.set_title(f"{title}\nSpearman {rho:.2f}")
    ax.set_xlabel("Mean PGA (g)")
axes[0].set_ylabel("Share grade $\\geq$4\n(stone masonry)")
axes[0].set_ylim(0, 1)
PL.save(fig, C.FIGURES_DIR / "05b_pga_vs_damage.png")
