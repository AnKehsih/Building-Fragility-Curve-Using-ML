"""
11b_ale_shap.py

Purpose : Post-hoc interpretation (added after the final test, Step 34; methods adopted from Borah et al. 2026,
          Nat. Commun. 17:5997). Development data only; no model is selected or scored, the frozen models
          and the final test are untouched. The two models that beat the fragility curves (ordinal logit and
          hybrid LightGBM, 09b_selected_final.csv) are fitted once on all development buildings (as in 11).
          A. Accumulated local effects (ALE) on P(grade >= 4): how the predicted severe-damage probability
             changes as one input moves within its observed range, from buildings that actually have values
             near each point (unlike the partial dependence of 11, which moves every building to every value
             and so creates combinations that do not occur, e.g. adobe houses at RC-typical PGA). Numeric inputs:
             age, storeys, plinth area, height, PGA (PGA also for the fragility curves). Two-input view as
             stratified ALE: the age effect within storeys 1 / 2 / 3, and the PGA effect within the main
             structural types (does one input's effect depend on the other?).
          B. SHAP for the hybrid (TreeSHAP, LightGBM's own pred_contrib): the trees start from the fragility
             curve, so their contributions say which attributes move a building above or below its type's
             curve. Per class the contributions are on the raw (log-score) scale; reported as the severity
             contrast mean(grades 4, 5) - mean(grades 1, 2, 3), which is additive (positive = pushes towards
             severe damage). Mean |contribution| per input, and mean contribution per category / flag value /
             numeric bin (direction). Additivity is checked against the model's raw scores.
          C. Association between inputs (why correlated inputs share credit in 09e): |Spearman| for numeric
             pairs, Cramer's V for categorical pairs (incl. the 0/1 flags), correlation ratio for numeric-
             categorical pairs; average-linkage clustering on 1 - association.
Inputs  : data/processed/* (as 08), results/tables/09b_selected_final.csv
Outputs : results/tables/11b_ale.csv, 11b_ale_stratified.csv, 11b_shap_importance.csv, 11b_shap_levels.csv,
          11b_association.csv, 11b_association_pairs.csv, 11b_run.json, 11b_console_output.txt (via redirect);
          results/figures/11b_ale.png, 11b_ale_stratified.png, 11b_shap.png, 11b_association.png
Run     : python code/11b_ale_shap.py --threads 8 > results/tables/11b_console_output.txt
          python code/11b_ale_shap.py --quick --threads 4     (smoke test)
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.cvrun import model_data
from utils.metrics import p_severe
from utils.models import SUPERSTRUCTURE, _lgbm_frame, _norm, make_model

T = C.TABLES_DIR
N_BINS, N_SAMPLE = 20, 100_000
ALE_FEATURES = ["age_building", "count_floors_pre_eq", "plinth_area_sq_ft", "height_ft_pre_eq", "pga"]
LABEL = {"fragility": "Fragility curves", "ordinal_logit": "Ordinal logit", "lightgbm": "Hybrid LightGBM"}
NAMES = {"age_building": "Age (years)", "count_floors_pre_eq": "Storeys", "plinth_area_sq_ft": "Plinth area (sq ft)",
         "height_ft_pre_eq": "Height (ft)", "pga": "PGA (g)"}
TYPES = ["stone_masonry", "mud_mortar_brick", "cement_mortar_masonry", "rc_engineered"]


def fit_models(src, threads, quick):
    df = model_data(src, "municipality")
    if quick:
        df = df.sample(30_000, random_state=C.SEED).reset_index(drop=True)
    s = pd.read_csv(T / "09b_selected_final.csv")
    y, g = df[C.TARGET].to_numpy(int), df[C.GROUP_COL].to_numpy()
    models = {"fragility": make_model("fragility", {}, "type_pga", threads, C.SEED).fit(df, y, g)}
    for r in s[(s.pga_source == src) & s.model.isin(["ordinal_logit", "lightgbm"])].itertuples():
        models[r.model] = make_model(r.model, json.loads(r.params), r.feature_set, threads, C.SEED, quick).fit(df, y, g)
    return df, models


def f_severe(model, X):
    return p_severe(_norm(model.predict_proba(X)))


# ---------------------------------------------------------------- A. ALE
def edges_for(x, col):
    x = x[~np.isnan(x)]
    if col == "count_floors_pre_eq":
        return np.arange(1, min(int(x.max()), 5) + 1, dtype=float)       # 1 -> 2 -> ... -> 5 storeys
    # 1st-99th percentile: the few extreme buildings (age 200, height 99 ft) would otherwise set the last interval
    return np.unique(np.quantile(x, np.linspace(0.01, 0.99, N_BINS + 1)))


def ale_1d(model, X, col, edges):
    """First-order ALE of P(grade >= 4) for `col` on `edges`, centred on the buildings' distribution.
    Returns DataFrame (x = edge, ale, n = buildings in the interval ending at this edge)."""
    x = X[col].to_numpy(float)
    ok = ~np.isnan(x) & (x >= edges[0]) & (x <= edges[-1])
    k = np.clip(np.searchsorted(edges, x, side="left") - 1, 0, len(edges) - 2)     # interval (e_k, e_k+1]
    idx = [np.flatnonzero(ok & (k == b)) for b in range(len(edges) - 1)]
    rows = np.concatenate(idx)
    lo, hi = X.iloc[rows].copy(), X.iloc[rows].copy()
    e_lo = np.concatenate([np.full(len(i), edges[b]) for b, i in enumerate(idx)])
    e_hi = np.concatenate([np.full(len(i), edges[b + 1]) for b, i in enumerate(idx)])
    for frame, v in ((lo, e_lo), (hi, e_hi)):
        frame[col] = v
        if col == "pga":
            frame["ln_pga"] = np.log(v)
    d = f_severe(model, hi) - f_severe(model, lo)
    n = np.array([len(i) for i in idx])
    delta = np.array([d[s:s + m].mean() if m else 0.0 for s, m in zip(np.r_[0, np.cumsum(n)[:-1]], n)])
    ale = np.r_[0.0, np.cumsum(delta)]
    mid = (ale[:-1] + ale[1:]) / 2
    ale = ale - (mid * n).sum() / max(n.sum(), 1)                            # centred: mean effect zero
    return pd.DataFrame({"x": edges, "ale": ale, "n": np.r_[0, n]})


# ---------------------------------------------------------------- B. SHAP (hybrid)
def shap_hybrid(m, X):
    """Severity contrast of TreeSHAP contributions: mean over grades 4-5 minus mean over grades 1-3."""
    Xl = _lgbm_frame(X, m.fs, m.categories_)
    raw_c = m.model_.predict(Xl, pred_contrib=True, num_iteration=m.best_iteration_)
    K, F = C.N_GRADES, Xl.shape[1]
    phi = np.asarray(raw_c).reshape(len(Xl), K, F + 1)                        # per class: F features + bias
    raw = m.model_.predict_proba(Xl, raw_score=True, num_iteration=m.best_iteration_)
    err = np.abs(phi.sum(2) - raw).max()
    sev = C.SEVERE_MIN_GRADE - 1
    contrast = phi[:, sev:, :].mean(1) - phi[:, :sev, :].mean(1)
    return pd.DataFrame(contrast[:, :F], columns=Xl.columns, index=X.index), float(err)


# Correlated inputs share credit, sometimes with opposite signs (e.g. structural type vs ground floor type,
# association 0.6-1.0, part C), so directions are read from group sums, which are additive and unaffected.
SHAP_GROUPS = {"Construction": ["structural_type", "foundation_type", "roof_type", "ground_floor_type",
                                "other_floor_type"] + SUPERSTRUCTURE,
               "Size": ["count_floors_pre_eq", "height_ft_pre_eq", "plinth_area_sq_ft"],
               "Age": ["age_building"],
               "Site and layout": ["land_surface_condition", "position", "plan_configuration"]}


def shap_groups(S, X):
    G = pd.DataFrame({g: S[[c for c in cols if c in S.columns]].sum(1) for g, cols in SHAP_GROUPS.items()})
    G["All attributes"] = S.sum(1)
    imp = pd.DataFrame({"group": G.columns, "mean_abs_shap": G.abs().mean().to_numpy(), "mean_shap": G.mean().to_numpy()})
    by_type = G.groupby(X.structural_type).mean().stack().rename("mean_shap").reset_index()
    by_type.columns = ["structural_type", "group", "mean_shap"]
    by_type["n"] = by_type.structural_type.map(X.structural_type.value_counts())
    return imp, by_type


def shap_levels(S, X):
    rows = []
    for c in S.columns:
        v = X[c]
        if c in SUPERSTRUCTURE:
            groups = v.map({0: "no", 1: "yes"})
        elif v.dtype.kind in "if" and v.nunique() > 12:
            groups = pd.qcut(v, 5, duplicates="drop").astype(str).replace({"nan": "missing"})
        else:
            groups = v.astype(str)
        for level, g in S[c].groupby(groups):
            rows.append({"feature": c, "level": level, "n": len(g), "mean_shap": g.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C. associations
def association_matrix(X, cols):
    from scipy.stats import spearmanr
    num = [c for c in cols if X[c].dtype.kind == "f" or (X[c].dtype.kind == "i" and c not in SUPERSTRUCTURE)]
    A = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

    def cramers_v(a, b):
        t = pd.crosstab(a, b).to_numpy(float)
        e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
        chi2 = ((t - e) ** 2 / e).sum()
        return np.sqrt(chi2 / t.sum() / max(1, min(t.shape) - 1))

    def eta(xn, cat):
        d = pd.DataFrame({"x": xn, "c": cat}).dropna()
        m = d.groupby("c").x.agg(["mean", "size"])
        return np.sqrt(((m["mean"] - d.x.mean()) ** 2 * m["size"]).sum() / ((d.x - d.x.mean()) ** 2).sum())

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in num and b in num:
                v = abs(spearmanr(X[a], X[b], nan_policy="omit")[0])
            elif a in num or b in num:
                n_, c_ = (a, b) if a in num else (b, a)
                v = eta(X[n_], X[c_].astype(str))
            else:
                v = cramers_v(X[a].astype(str), X[b].astype(str))
            A.loc[a, b] = A.loc[b, a] = v
    return A


# ---------------------------------------------------------------- figures
def figures(ale, strat, gimp, gtype, A, order, sfx):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from utils.plotting import (CATEGORICAL, FULL_WIDTH_IN, GRADE_COLORS, MUTED, SURFACE, save, set_style)
    set_style()
    src = C.SHAKEMAP_PRIMARY
    a = ale[ale.pga_source == src]
    fig, axes = plt.subplots(1, 5, figsize=(FULL_WIDTH_IN, 2.0))
    for ax, col in zip(axes, ALE_FEATURES):
        ax.axhline(0, color=MUTED, lw=0.8)
        for i, m in enumerate(["fragility", "ordinal_logit", "lightgbm"]):
            g = a[(a.feature == col) & (a.model == m)]
            if len(g):
                ax.plot(g.x, 100 * g.ale, color=CATEGORICAL[i], lw=1.4, ls="--" if m == "fragility" else "-",
                        label=LABEL[m])
        ax.set(xlabel=NAMES[col])
        if col == "plinth_area_sq_ft":
            ax.set_xscale("log")
    axes[0].set_ylabel("Δ P(grade ≥ 4), points")
    h, l = axes[-1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, C.FIGURES_DIR / f"11b_ale{sfx}.png")              # caption: ALE, preferred map, development data

    s = strat[(strat.pga_source == src) & (strat.model == "lightgbm")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.4))
    for j, st in enumerate(["1", "2", "3"]):                     # storeys are ordered: one-hue ramp
        g = s[(s.feature == "age_building") & (s.stratum == st)]
        axes[0].plot(g.x, 100 * g.ale, color=GRADE_COLORS[2 * j], lw=1.4, label=f"{st} storey" + ("s" if st != "1" else ""))
    axes[0].set(xlabel="Age (years)", ylabel="Δ P(grade ≥ 4), points", title="Hybrid: age effect by storeys")
    axes[0].legend(loc="lower right")
    for i, t in enumerate(TYPES):
        g = s[(s.feature == "pga") & (s.stratum == t)]
        axes[1].plot(g.x, 100 * g.ale, color=CATEGORICAL[i], lw=1.4, label=t.replace("_", " ").replace("rc ", "RC "))
    axes[1].set(xlabel="PGA (g)", title="Hybrid: PGA effect by structural type")
    axes[1].legend(loc="lower right", fontsize=6)
    for ax in axes:
        ax.axhline(0, color=MUTED, lw=0.8)
    fig.tight_layout()
    save(fig, C.FIGURES_DIR / f"11b_ale_stratified{sfx}.png")

    groups = list(SHAP_GROUPS)                                  # fixed colour slots 0..3
    gi = gimp[(gimp.pga_source == src) & gimp.group.isin(groups)].set_index("group").loc[groups[::-1]]
    gt = gtype[(gtype.pga_source == src) & gtype.group.isin(groups)]
    types = (gt[gt.group == "Construction"].sort_values("mean_shap").structural_type.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.8), gridspec_kw={"width_ratios": [1, 1.5]})
    axes[0].barh(gi.index, gi.mean_abs_shap, color=[CATEGORICAL[groups.index(g)] for g in gi.index], height=0.6)
    axes[0].set(xlabel="Mean |SHAP|, severity contrast", title="What moves buildings off their curve")
    axes[0].grid(axis="y", visible=False)
    axes[1].axvline(0, color=MUTED, lw=0.8)
    for i, g in enumerate(groups):
        d = gt[gt.group == g].set_index("structural_type").loc[types]
        axes[1].scatter(d.mean_shap, np.arange(len(types)) + (i - 1.5) * 0.15, s=16, color=CATEGORICAL[i],
                        edgecolor=SURFACE, lw=0.6, label=g, zorder=3)
    axes[1].set_yticks(range(len(types)), [t.replace("_", " ").replace("rc ", "RC ") for t in types], fontsize=6)
    axes[1].set(xlabel="Mean SHAP (+ = towards severe)", title="Net direction by structural type")  # vs average building
    axes[1].grid(axis="y", visible=False)
    axes[1].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    save(fig, C.FIGURES_DIR / f"11b_shap{sfx}.png")             # caption: hybrid, preferred map, 100k buildings

    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, GRADE_COLORS[0], GRADE_COLORS[2], GRADE_COLORS[4]])
    B = A.loc[order, order]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 5.6))
    m = ax.imshow(B.to_numpy(), cmap=cmap, vmin=0, vmax=1)
    lab = [c.replace("has_superstructure_", "super: ").replace("_", " ") for c in order]
    ax.set_xticks(range(len(order)), lab, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)), lab, fontsize=6)
    ax.grid(False)
    fig.colorbar(m, ax=ax, shrink=0.6, label="Association (|Spearman|, Cramér's V, correlation ratio)")
    fig.tight_layout()
    save(fig, C.FIGURES_DIR / f"11b_association{sfx}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="smoke test: 30k training rows, 5k sample")
    a = p.parse_args()
    sfx = "_quick" if a.quick else ""
    n_sample = 5_000 if a.quick else N_SAMPLE
    t0, meta = time.time(), {"args": vars(a), "python": platform.python_version(),
                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    ale, strat, imp, lev, gimp, gtype = [], [], [], [], [], []
    for src in C.SHAKEMAP_SOURCES:
        df, models = fit_models(src, a.threads, a.quick)
        X = df.sample(min(n_sample, len(df)), random_state=C.SEED).reset_index(drop=True)
        for col in ALE_FEATURES:
            edges = edges_for(df[col].to_numpy(float), col)
            for name, m in models.items():
                if name == "fragility" and col != "pga":
                    continue
                ale.append(ale_1d(m, X, col, edges).assign(pga_source=src, model=name, feature=col))
        for name in ("ordinal_logit", "lightgbm"):
            m = models[name]
            for st in (1, 2, 3):
                sub = X[X.count_floors_pre_eq == st].reset_index(drop=True)
                e = edges_for(sub.age_building.to_numpy(float), "age_building")
                strat.append(ale_1d(m, sub, "age_building", e).assign(pga_source=src, model=name,
                                                                       feature="age_building", stratum=str(st)))
            for t in TYPES:
                sub = X[X.structural_type == t].reset_index(drop=True)
                e = edges_for(sub.pga.to_numpy(float), "pga")
                if len(sub) >= 200 and len(e) >= 3:
                    strat.append(ale_1d(m, sub, "pga", e).assign(pga_source=src, model=name, feature="pga", stratum=t))
        S, err = shap_hybrid(models["lightgbm"], X)
        meta[f"shap_additivity_max_error_{src}"] = err
        print(f"{src}: SHAP additivity check, max |sum of contributions - raw score| = {err:.2e}", flush=True)
        imp.append(pd.DataFrame({"feature": S.columns, "mean_abs_shap": S.abs().mean().to_numpy(),
                                 "mean_shap": S.mean().to_numpy()}).assign(pga_source=src))
        lev.append(shap_levels(S, X).assign(pga_source=src))
        gi, gt = shap_groups(S, X)
        gimp.append(gi.assign(pga_source=src))
        gtype.append(gt.assign(pga_source=src))
        if src == C.SHAKEMAP_PRIMARY:
            cols = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + ["structural_type"] + SUPERSTRUCTURE + ["ln_pga"]
            A = association_matrix(X, cols)
    ale, strat, imp, lev = pd.concat(ale), pd.concat(strat), pd.concat(imp), pd.concat(lev)
    gimp, gtype = pd.concat(gimp), pd.concat(gtype)

    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    D = 1 - A.to_numpy()
    np.fill_diagonal(D, 0)
    order = list(A.index[leaves_list(linkage(squareform((D + D.T) / 2, checks=False), "average"))])
    pairs = (A.where(np.triu(np.ones(A.shape, bool), 1)).stack().rename("association").reset_index()
             .rename(columns={"level_0": "a", "level_1": "b"}).sort_values("association", ascending=False))
    for d, name in ((ale, "ale"), (strat, "ale_stratified"), (imp, "shap_importance"), (lev, "shap_levels"),
                    (gimp, "shap_groups"), (gtype, "shap_groups_by_type"),
                    (A, "association"), (pairs, "association_pairs")):
        d.to_csv(T / f"11b_{name}{sfx}.csv", index=(name == "association"))

    pd.set_option("display.width", 220)
    print("\nA. ALE of P(grade >= 4) (percentage points; centred). Range = max - min over the input's observed range:")
    rng = ale.groupby(["pga_source", "feature", "model"]).ale.agg(lambda v: 100 * (v.max() - v.min())).unstack()
    print(rng.round(1).to_string())
    for src in C.SHAKEMAP_SOURCES:
        print(f"\n{src}, hybrid, ALE at the interval edges (points):")
        for col in ALE_FEATURES[:4]:
            g = ale[(ale.pga_source == src) & (ale.model == "lightgbm") & (ale.feature == col)]
            print(f"  {col:20s} " + "  ".join(f"{x:g}:{100 * v:+.1f}" for x, v in zip(g.x, g.ale)))
    print("\nStratified ALE ranges (points): age within storeys; PGA within structural types")
    print(strat.groupby(["pga_source", "model", "feature", "stratum"]).ale.agg(
        lambda v: 100 * (v.max() - v.min())).unstack("stratum").round(1).to_string())
    print("\nB. Hybrid SHAP (severity contrast: mean raw score of grades 4-5 minus grades 1-3; + = towards severe)")
    print(imp.pivot(index="feature", columns="pga_source", values="mean_abs_shap")
          .sort_values(C.SHAKEMAP_PRIMARY, ascending=False).round(3).to_string())
    print("\nGrouped SHAP (sums over correlated inputs; read directions from these):")
    print(gimp.pivot(index="group", columns="pga_source", values=["mean_abs_shap", "mean_shap"]).round(3).to_string())
    print("\nMean grouped SHAP by structural type (relative to the average building's tree correction;"
          " + = towards severe):")
    print(gtype.pivot_table(index=["pga_source", "structural_type", "n"], columns="group", values="mean_shap")
          .round(3).to_string())
    print("\nDirection by level, single inputs (preferred map; >= 500 buildings; see grouping caveat):")
    print(lev[(lev.pga_source == C.SHAKEMAP_PRIMARY) & (lev.n >= 500)].round(3).to_string(index=False))
    print("\nC. Strongest associations between inputs (development sample):")
    print(pairs.head(15).round(2).to_string(index=False))
    print("Clustering order:", ", ".join(order))
    meta |= {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - t0) / 60, 1)}
    (T / f"11b_run{sfx}.json").write_text(json.dumps(meta, indent=2))
    figures(ale, strat, gimp, gtype, A, order, sfx)


if __name__ == "__main__":
    main()
