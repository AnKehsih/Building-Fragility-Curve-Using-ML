"""
11_fragility_comparison.py

Purpose : Graduate analysis 3 (CONTEXT 6): what do the trained models imply as fragility curves? For each
          structural type, the probability of reaching grade k (k = 2..5) against PGA, from averaged
          predictions (partial dependence): take the development buildings of that type (up to 20,000,
          sampled), set every building's PGA to a grid value, predict, average P(grade >= k). Plotted on
          the same axes as the lognormal fragility curves and the observed exceedance shares (buildings of
          the type binned by municipality PGA).
          Models (both maps), fitted ONCE on all development buildings (interpretation only; nothing is
          scored, the final-test district is not used):
            fragility       lognormal curves per type (as 07)
            ordinal logit   09b-selected configuration (linear in ln PGA)
            hybrid LightGBM 09b-selected (boosted from the fragility curve; trees do not see PGA, so its
                            PGA shape comes from the curve and the trees shift it by building attributes)
            plain LightGBM  08b-selected configuration, full inputs (learns PGA itself; Step 19: PGA acts
                            as a municipality identifier)
          Each model curve is summarised as a lognormal (theta_k, beta) by a least-squares line in probit
          space over the type's observed PGA range (5th-95th percentile of its buildings), with the number
          of decreases of the curve along the grid (a fragility curve should not fall as shaking rises)
          and its mean absolute distance from the fragility curve over that range.
          Caveat: partial dependence moves each building to PGA values it never experienced; outside a
          type's observed range the curves are extrapolations.
Inputs  : data/processed/* (as 08), results/tables/09b_selected_final.csv, 08b_selected.csv
Outputs : results/tables/11_curves.csv, 11_observed.csv, 11_implied_params.csv, 11_run.json,
          11_console_output.txt (via redirect); results/figures/11_severe_by_type_<map>.png,
          11_stone_masonry_grades.png
Run     : python code/11_fragility_comparison.py --threads 8 > results/tables/11_console_output.txt
          python code/11_fragility_comparison.py --quick --threads 4      (smoke test)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.cvrun import model_data
from utils.fragility import FLAT_BETA
from utils.models import GRIDS, _norm, config_name, make_model

T = C.TABLES_DIR
MODELS = ("fragility", "ordinal_logit", "lightgbm", "lightgbm_plain")     # fixed colour slots 0..3
LABEL = {"fragility": "Fragility curves", "ordinal_logit": "Ordinal logit", "lightgbm": "Hybrid LightGBM",
         "lightgbm_plain": "Plain LightGBM"}
STYLE = {"fragility": "--", "ordinal_logit": "-", "lightgbm": "-", "lightgbm_plain": "-"}
TYPES = ["stone_masonry", "cement_mortar_masonry", "timber_bamboo", "rc_non_engineered", "mixed_strong_weak",
         "rc_engineered", "mud_mortar_brick", "adobe_mud"]                 # "other" (981 buildings) left out
N_PER_TYPE, N_GRID, N_OBS_BINS = 20_000, 30, 8
EXCEED = range(2, C.N_GRADES + 1)


def specs():
    """{model: (make_model name, params, feature set)} per map."""
    s9 = pd.read_csv(T / "09b_selected_final.csv")
    s8 = pd.read_csv(T / "08b_selected.csv")
    by_name = {(m, config_name(p)): p for m, grid in GRIDS.items() for p in grid}
    out = {}
    for src in C.SHAKEMAP_SOURCES:
        sel = {r.model: (r.model, json.loads(r.params), r.feature_set) for r in s9[s9.pga_source == src].itertuples()}
        cfg8 = s8[(s8.scheme == "municipality") & (s8.pga_source == src) & (s8.model == "lightgbm")].config.iloc[0]
        out[src] = {"fragility": ("fragility", {}, "type_pga"), "ordinal_logit": sel["ordinal_logit"],
                    "lightgbm": sel["lightgbm"], "lightgbm_plain": ("lightgbm", by_name["lightgbm", cfg8], "full")}
    return out


def exceedance(P):
    """P (n, 5) -> P(grade >= k) for k = 2..5, shape (n, 4)."""
    return 1 - np.cumsum(P, axis=1)[:, :-1]


def model_curves(models, df, grid, n_per_type):
    rows = []
    for c in TYPES:
        sub = df[df.structural_type == c]
        s = sub.sample(min(len(sub), n_per_type), random_state=C.SEED)
        x = pd.concat([s.assign(pga=g, ln_pga=np.log(g)) for g in grid], ignore_index=True)
        for name, m in models.items():
            ex = exceedance(_norm(m.predict_proba(x))).reshape(len(grid), len(s), -1).mean(axis=1)
            rows += [{"structural_type": c, "model": name, "pga": g, "k": k, "p": ex[i, j]}
                     for i, g in enumerate(grid) for j, k in enumerate(EXCEED)]
    return pd.DataFrame(rows)


def observed(df):
    rows = []
    for c in TYPES:
        sub = df[df.structural_type == c]
        b = pd.qcut(sub.pga, N_OBS_BINS, duplicates="drop")
        for _, g in sub.groupby(b, observed=True):
            y = g[C.TARGET].to_numpy(int)
            rows += [{"structural_type": c, "pga": g.pga.mean(), "n": len(g), "k": k, "p": (y >= k).mean()}
                     for k in EXCEED]
    return pd.DataFrame(rows)


def implied(curves, df):
    """Lognormal summary of each curve over the type's observed PGA range."""
    rows = []
    for (c, name, k), g in curves.groupby(["structural_type", "model", "k"]):
        lo, hi = df.loc[df.structural_type == c, "pga"].quantile([0.05, 0.95])
        r = g[(g.pga >= lo) & (g.pga <= hi)]
        if len(r) < 3:                                   # range narrower than the grid spacing (quick runs)
            continue
        z, x = norm.ppf(np.clip(r.p, 1e-4, 1 - 1e-4)), np.log(r.pga)
        b, a = np.polyfit(x, z, 1)
        f = curves[(curves.structural_type == c) & (curves.model == "fragility") & (curves.k == k)]
        f = f[(f.pga >= lo) & (f.pga <= hi)]
        rows.append({"structural_type": c, "model": name, "k": k, "pga_lo": lo, "pga_hi": hi,
                     "theta": np.exp(-a / b) if b > 0 else np.nan, "beta": 1 / b if b > 0 else np.nan,
                     "n_decreases": int((np.diff(g.sort_values("pga").p) < -0.002).sum()),
                     "mad_vs_fragility": np.abs(r.p.to_numpy() - f.p.to_numpy()).mean(),
                     "p_at_lo": r.p.iloc[0], "p_at_hi": r.p.iloc[-1]})
    return pd.DataFrame(rows)


def pretty(c):
    return c.replace("_", " ").replace("rc ", "RC ").capitalize().replace("Rc ", "RC ")


def figures(curves, obs, sfx):
    import matplotlib.pyplot as plt
    from utils.plotting import CATEGORICAL, FULL_WIDTH_IN, GRADE_COLORS, INK, save, set_style
    set_style()
    titles = {"preferred": "preferred ShakeMap", "neic_v9": "NEIC v9 ShakeMap"}
    for src in C.SHAKEMAP_SOURCES:
        cu, ob = curves[(curves.pga_source == src) & (curves.k == 4)], obs[(obs.pga_source == src) & (obs.k == 4)]
        fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH_IN, 3.9), sharex=True, sharey=True)
        for ax, c in zip(axes.flat, TYPES):
            for i, m in enumerate(MODELS):
                g = cu[(cu.structural_type == c) & (cu.model == m)]
                ax.plot(g.pga, g.p, ls=STYLE[m], color=CATEGORICAL[i], lw=1.4, label=LABEL[m])
            o = ob[ob.structural_type == c]
            ax.scatter(o.pga, o.p, s=12, color=INK, zorder=4, label="Observed (binned)")
            ax.set_title(pretty(c), fontsize=7)
            ax.set(ylim=(0, 1))
        for ax in axes[1]:
            ax.set_xlabel("PGA (g)")
        for ax in axes[:, 0]:
            ax.set_ylabel("P(grade ≥ 4)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Severe damage against PGA by structural type, {titles[src]}", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        save(fig, C.FIGURES_DIR / f"11_severe_by_type_{src}{sfx}.png")

    # stone masonry (80% of buildings): all four thresholds; colour = grade (ordinal ramp), style = model
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.7), sharey=True)
    for ax, src in zip(axes, C.SHAKEMAP_SOURCES):
        cu = curves[(curves.pga_source == src) & (curves.structural_type == "stone_masonry")]
        ob = obs[(obs.pga_source == src) & (obs.structural_type == "stone_masonry")]
        ends = []
        for k in EXCEED:
            for m, ls in (("fragility", "--"), ("lightgbm", "-")):
                g = cu[(cu.model == m) & (cu.k == k)]
                ax.plot(g.pga, g.p, ls=ls, color=GRADE_COLORS[k - 1], lw=1.4)
            o = ob[ob.k == k]
            ax.scatter(o.pga, o.p, s=12, color=GRADE_COLORS[k - 1], edgecolor=INK, lw=0.5, zorder=4)
            ends.append((k, cu[(cu.model == "lightgbm") & (cu.k == k)].iloc[-1]))
        y_prev = np.inf                                   # end labels, pushed apart so they never overlap
        for k, g in ends:
            y_prev = min(g.p, y_prev - 0.06)
            ax.annotate(f"≥ {k}", (g.pga, y_prev), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=7)
        ax.set(ylim=(0, 1), xlabel="PGA (g)", title=f"Stone masonry, {titles[src]}")
    axes[0].set_ylabel("P(grade ≥ k)")
    axes[0].plot([], [], ls="--", color=INK, label="Fragility curves")
    axes[0].plot([], [], ls="-", color=INK, label="Hybrid LightGBM")
    axes[0].scatter([], [], s=12, color="white", edgecolor=INK, lw=0.5, label="Observed (binned; colour = grade k)")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, C.FIGURES_DIR / f"11_stone_masonry_grades{sfx}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="smoke test: 30k training rows, 2k per type, 8 PGA values")
    a = p.parse_args()
    sfx = "_quick" if a.quick else ""
    t0, meta = time.time(), {"args": vars(a), "python": platform.python_version(),
                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    curves, obs, imp = [], [], []
    for src, spec in specs().items():
        df = model_data(src, "municipality")
        if a.quick:
            df = df.sample(30_000, random_state=C.SEED).reset_index(drop=True)
        y, groups = df[C.TARGET].to_numpy(int), df[C.GROUP_COL].to_numpy()
        models = {}
        for name, (mname, params, fs) in spec.items():
            t = time.time()
            models[name] = make_model(mname, params, fs, a.threads, C.SEED, a.quick).fit(df, y, groups)
            print(f"{src}: fitted {name} on {len(df):,} buildings ({time.time() - t:.0f} s)", flush=True)
        grid = np.geomspace(df.pga.min(), df.pga.max(), 8 if a.quick else N_GRID)
        cu = model_curves(models, df, grid, 2_000 if a.quick else N_PER_TYPE).assign(pga_source=src)
        curves.append(cu)
        obs.append(observed(df).assign(pga_source=src))
        imp.append(implied(cu, df).assign(pga_source=src))
    curves, obs, imp = pd.concat(curves), pd.concat(obs), pd.concat(imp)
    curves.to_csv(T / f"11_curves{sfx}.csv", index=False)
    obs.to_csv(T / f"11_observed{sfx}.csv", index=False)
    imp.to_csv(T / f"11_implied_params{sfx}.csv", index=False)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round((time.time() - t0) / 60, 1)
    (T / f"11_run{sfx}.json").write_text(json.dumps(meta, indent=2, default=str))

    pd.set_option("display.width", 220)
    print("\nImplied lognormal parameters per model (least-squares line in probit space over the type's observed")
    print("PGA range, 5th-95th percentile). theta: median PGA (g) for reaching grade k; beta: dispersion;")
    print("n_decreases: grid steps where the curve falls by > 0.002; mad_vs_fragility: mean |difference| from the")
    print("fragility curve over the observed range. The fragility rows reproduce the 07 parameters (sanity check).")
    # beta > FLAT_BETA: the curve barely rises over the observed range, so theta is a meaningless extrapolation
    imp["theta/beta"] = [f"flat (beta {b:.1f})" if b > FLAT_BETA else f"{t:.3f} / {b:.2f}"
                         for t, b in zip(imp.theta, imp.beta)]
    for k in EXCEED:
        s = imp[imp.k == k]
        w = s.pivot(index=["pga_source", "structural_type"], columns="model", values="theta/beta")[list(MODELS)]
        w = w.join(s.pivot(index=["pga_source", "structural_type"], columns="model", values="mad_vs_fragility")
                   [list(MODELS[1:])].round(3).add_prefix("mad_"))
        w = w.join(s.pivot(index=["pga_source", "structural_type"], columns="model", values="n_decreases")
                   [list(MODELS[1:])].add_prefix("dec_"))
        print(f"\nGrade >= {k}: theta (g) / beta per model; mad_*: mean |difference| from the fragility curve;"
              f" dec_*: decreases")
        print(w.to_string())
    figures(curves, obs, sfx)


if __name__ == "__main__":
    main()
