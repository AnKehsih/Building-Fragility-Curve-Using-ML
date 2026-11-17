"""
12_failure_audit.py

Purpose : Graduate analysis 4 (CONTEXT 6): where do the models fail, how noisy are the labels, and how much
          would post-earthquake fields inflate the scores?
          A. Error breakdown on new districts: the leave-one-district-out out-of-fold predictions of 09c
             (fragility curves, ordinal logit, hybrid LightGBM; municipality PGA; both maps) broken down by
             district, storeys, age band, slope (land surface condition) and structural type: observed and
             predicted severe share, RPS per model, paired difference vs the fragility curves.
          B. Where the remaining error sits: the severe-damage residual (outcome minus predicted P(grade >= 4))
             split exactly into district, municipality-within-district, ward-within-municipality and
             within-ward parts (nested sums of squares; Brier = sum of the four). The same split of the
             outcome itself shows how much of the variation is spatial at all.
          C. Label noise (no surveyor IDs exist in the data): "twins" are buildings in the same ward with
             identical values of every input (all 22 attribute columns). They share shaking, location and
             recorded construction, so differences in their grades are surveyor disagreement plus what the
             survey does not record. Reported: grade agreement between twins, and a noise floor for RPS: the
             expected RPS of predicting each twin group's own grade distribution exactly, estimated without
             bias as (1/4) sum_k n/(n-1) G_k (1 - G_k). No model using these inputs (even with the ward
             known) can beat it on these buildings; compared with the models' RPS on the same buildings.
          D. LEAKAGE DEMONSTRATION (clearly labelled; CONTEXT 3.3, the only place post-earthquake columns are
             used): the 08b-selected plain LightGBM refitted with the four post-earthquake columns added
             (floors and height after the earthquake, condition, proposed technical solution), municipality
             folds seed 678, preferred map, compared fold by fold with the same model without them (08).
Inputs  : data/processed/* (incl. buildings_posteq.parquet for D only), results/oof/09c_*.parquet,
          results/tables/09b_selected_final.csv, 08b_selected.csv, 08_cv_metrics_main.csv, 07_cv_metrics.csv
Outputs : results/tables/12_breakdown.csv, 12_residual_decomposition.csv, 12_twins.csv,
          12_leakage_demo_metrics.csv, 12_leakage_demo_summary.csv, 12_run.json, 12_console_output.txt
          (via redirect); results/figures/12_rps_by_subgroup.png
Run     : python code/12_failure_audit.py --n_jobs 5 --threads 4 > results/tables/12_console_output.txt
          python code/12_failure_audit.py --quick --n_jobs 2 --threads 2    (smoke test)
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
from utils.cv import fold_ids
from utils.cvrun import OOF_DIR, model_data
from utils.data import load_posteq_for_leakage_demo
from utils.metrics import evaluate, p_severe, rps
from utils.models import FEATURE_SETS, GRIDS, SUPERSTRUCTURE, config_name, fit_predict

T = C.TABLES_DIR
MODELS = ("fragility", "ordinal_logit", "lightgbm")                     # fixed colour slots 0..2
LABEL = {"fragility": "Fragility curves", "ordinal_logit": "Ordinal logit", "lightgbm": "Hybrid LightGBM"}
INPUTS = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + SUPERSTRUCTURE   # the 22 attribute columns
AGE_BANDS = [0, 10, 20, 40, 80, np.inf]
LEAK = "LEAKAGE_DEMO_posteq"


def oof_names():
    s = pd.read_csv(T / "09b_selected_final.csv")
    out = {}
    for src in C.SHAKEMAP_SOURCES:
        out[src, "fragility"] = f"09c_fragility_na_{src}_municipality_type_pga"
        for r in s[(s.pga_source == src) & s.model.isin(MODELS)].itertuples():
            out[src, r.model] = f"09c_{r.model}_{r.config}_{src}_municipality_{r.feature_set}"
    return out


def subgroups(df):
    """Building attributes used for the breakdown (labels as shown in tables and figures)."""
    floors = df.count_floors_pre_eq.clip(upper=4).astype(int).astype(str).replace({"4": "4+"})
    age = pd.cut(df.age_building, AGE_BANDS, right=False,
                 labels=["0-9", "10-19", "20-39", "40-79", "80+"]).astype(str).replace({"nan": "missing"})
    return pd.DataFrame({"district": df.district_id.map(C.DISTRICTS), "storeys": floors, "age_band": age,
                         "slope": df.land_surface_condition, "structural_type": df.structural_type})


def lodo_frame(src, quick):
    """Development buildings with grade, subgroups and the three models' LODO predictions."""
    df = model_data(src, "municipality")
    base = pd.concat([df[["building_id", C.TARGET, C.GROUP_COL, "ward_id"] + INPUTS], subgroups(df)], axis=1)
    if quick:
        base = base.sample(50_000, random_state=C.SEED)
    y = base[C.TARGET].to_numpy(int)
    base["y_sev"] = (y >= C.SEVERE_MIN_GRADE).astype(float)
    for (s, model), name in oof_names().items():
        if s != src:
            continue
        o = pd.read_parquet(OOF_DIR / f"{name}.parquet").set_index("building_id").loc[base.building_id]
        P = o[[f"p{g}" for g in C.GRADES]].to_numpy(float)
        base[f"rps_{model}"], base[f"p_{model}"] = rps(P, y), p_severe(P)
    return base.reset_index(drop=True)


# ---------------------------------------------------------------- A. breakdown
def breakdown(b, src):
    rows = []
    for var in ["district", "storeys", "age_band", "slope", "structural_type"]:
        for level, g in b.groupby(var):
            r = {"pga_source": src, "variable": var, "level": level, "n": len(g), "share": len(g) / len(b),
                 "observed_severe": g.y_sev.mean()}
            for m in MODELS:
                r[f"pred_severe_{m}"] = g[f"p_{m}"].mean()
                r[f"rps_{m}"] = g[f"rps_{m}"].mean()
            for m in MODELS[1:]:
                r[f"d_rps_{m}"] = r[f"rps_{m}"] - r["rps_fragility"]
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B. residual decomposition
def nested_ss(v, d, m, w):
    """Split mean(v^2) into district / municipality-in-district / ward-in-municipality / within-ward parts."""
    f = pd.DataFrame({"v": v, "d": d, "m": m, "w": w})
    md = f.groupby("d").v.transform("mean")
    mm = f.groupby("m").v.transform("mean")
    mw = f.groupby("w").v.transform("mean")
    parts = {"district": md ** 2, "municipality": (mm - md) ** 2, "ward": (mw - mm) ** 2, "within_ward": (f.v - mw) ** 2}
    return {k: x.mean() for k, x in parts.items()}


def decomposition(b, src):
    rows = []
    d, m, w = b.district, b[C.GROUP_COL], b.ward_id
    out = nested_ss(b.y_sev - b.y_sev.mean(), d, m, w)
    rows.append({"pga_source": src, "quantity": "outcome (variance of grade >= 4)"} | out | {"total": sum(out.values())})
    for mod in MODELS:
        out = nested_ss(b.y_sev - b[f"p_{mod}"], d, m, w)
        rows.append({"pga_source": src, "quantity": f"residual, {mod} (Brier)"} | out | {"total": sum(out.values())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C. twins
def twins(b, src):
    key = ["ward_id"] + INPUTS
    g = b.assign(**{c: b[c].astype(str) for c in INPUTS}).groupby(key, sort=False, dropna=False)
    b = b.assign(gid=g.ngroup(), gn=g[C.TARGET].transform("size"))
    t = b[b.gn >= 2]
    y = t[C.TARGET].to_numpy(int)
    # pairwise agreement within groups, from per-group grade counts
    cnt = pd.crosstab(t.gid, y).reindex(columns=C.GRADES, fill_value=0).to_numpy(float)
    n = cnt.sum(1)
    pairs = n * (n - 1) / 2
    same = (cnt * (cnt - 1) / 2).sum(1)
    grades = np.array(C.GRADES, float)
    diff_sum = sum(cnt[:, i] * cnt[:, j] * abs(grades[i] - grades[j]) for i in range(5) for j in range(i + 1, 5))
    far = sum(cnt[:, i] * cnt[:, j] for i in range(5) for j in range(i + 1, 5) if grades[j] - grades[i] >= 2)
    G = np.cumsum(cnt, 1)[:, :-1] / n[:, None]                          # within-group cumulative shares
    floor = (n / (n - 1))[:, None] * G * (1 - G)
    floor_rps = (floor.sum(1) / (C.N_GRADES - 1) * n).sum() / n.sum()   # building-weighted
    n_lo, n_sev = cnt[:, :C.SEVERE_MIN_GRADE - 1].sum(1), cnt[:, C.SEVERE_MIN_GRADE - 1:].sum(1)
    same_sev = (n_lo * (n_lo - 1) / 2 + n_sev * (n_sev - 1) / 2).sum()
    row = {"pga_source": src, "buildings_in_twin_groups": len(t), "share_of_buildings": len(t) / len(b),
           "twin_groups": len(n), "median_group_size": float(np.median(n)),
           "pair_same_grade": same.sum() / pairs.sum(), "pair_mean_abs_diff": diff_sum.sum() / pairs.sum(),
           "pair_diff_2plus": far.sum() / pairs.sum(), "pair_same_severe": same_sev / pairs.sum(),
           "noise_floor_rps": floor_rps}
    for m in MODELS:
        row[f"rps_{m}_same_buildings"] = t[f"rps_{m}"].mean()
    row["rps_all_buildings_lightgbm"] = b.rps_lightgbm.mean()
    return row


# ---------------------------------------------------------------- D. leakage demonstration
def leakage_task(fold, df, f, params, threads, quick):
    tr, va = np.flatnonzero(f != fold), np.flatnonzero(f == fold)
    if quick:
        rng = np.random.default_rng(0)
        tr, va = np.sort(rng.choice(tr, 30000, replace=False)), np.sort(rng.choice(va, 10000, replace=False))
    y = df[C.TARGET].to_numpy(int)
    P, info = fit_predict("lightgbm", params, LEAK, df.iloc[tr], df.iloc[va], y[tr], threads,
                          seed=C.SEED * 10 + fold, groups=df[C.GROUP_COL].to_numpy()[tr], quick=quick)
    return {"fold": fold, "seed": C.SEED, "n": len(va)} | evaluate(P, y[va]) | {"best_iteration": info["best_iteration"]}


def leakage_demo(a):
    from joblib import Parallel, delayed
    src = C.SHAKEMAP_PRIMARY
    s8 = pd.read_csv(T / "08b_selected.csv")
    cfg = s8[(s8.scheme == "municipality") & (s8.pga_source == src) & (s8.model == "lightgbm")].config.iloc[0]
    params = {(m, config_name(p)): p for m, g in GRIDS.items() for p in g}["lightgbm", cfg]
    df = model_data(src, "municipality").merge(load_posteq_for_leakage_demo(), on="building_id", validate="1:1")
    assert set(FEATURE_SETS[LEAK]["categorical"]) <= set(df.columns)
    f = fold_ids(df, "municipality", C.SEED)
    rows = Parallel(n_jobs=a.n_jobs)(delayed(leakage_task)(k, df, f, params, a.threads, a.quick)
                                     for k in range(1 if a.quick else C.N_FOLDS))
    leak = pd.DataFrame(rows).assign(model="lightgbm", config=cfg, feature_set=LEAK, pga_source=src)
    m8 = pd.read_csv(T / "08_cv_metrics_main.csv")
    base = m8[(m8.model == "lightgbm") & (m8.config == cfg) & (m8.feature_set == "full") & (m8.pga_source == src)
              & (m8.pga_level == "municipality") & (m8.scheme == "municipality") & (m8.seed == C.SEED)]
    b7 = pd.read_csv(T / "07_cv_metrics.csv")
    frag = b7[(b7.model == f"fragility_{src}_municipality") & (b7.scheme == "municipality") & (b7.seed == C.SEED)]
    metrics = ["rps", "brier_severe", "log_loss", "qwk", "macro_f1", "accuracy"]
    rows = []
    for label, d in (("fragility curves (pre-earthquake: type + PGA)", frag),
                     ("LightGBM, pre-earthquake inputs (as 08)", base),
                     ("LightGBM + post-earthquake columns (LEAKAGE)", leak)):
        d = d[d.fold.isin(leak.fold)]
        rows.append({"model": label, "n_folds": len(d)} | {m: d[m].mean() for m in metrics})
    # how directly single post-earthquake values give away the grade (development buildings)
    pe = df[["count_floors_post_eq", "condition_post_eq", C.TARGET]]
    facts = {"zero_floors_after": int((pe.count_floors_post_eq == 0).sum()),
             "zero_floors_after_grade5_share": float((pe[pe.count_floors_post_eq == 0][C.TARGET] == 5).mean()),
             "not_damaged_grade1_share": float((pe[pe.condition_post_eq == "Not damaged"][C.TARGET] == 1).mean())}
    return leak, pd.DataFrame(rows), facts


# ---------------------------------------------------------------- figure
def figure(bd, sfx):
    """One dot plot: RPS difference vs the fragility curves per subgroup (shared axis, zero = no gain)."""
    import matplotlib.pyplot as plt
    from utils.plotting import CATEGORICAL, COL_WIDTH_IN, INK, MUTED, SURFACE, save, set_style
    set_style()
    d = bd[(bd.pga_source == C.SHAKEMAP_PRIMARY) & (bd.share >= 0.005)]
    sections = [("storeys", "Storeys"), ("age_band", "Age (years)"), ("slope", "Slope"),
                ("structural_type", "Structural type")]
    labels, ys, rows, y = [], [], [], 0
    for var, title in sections:
        labels.append(title), ys.append(y)
        y -= 1
        g = d[d.variable == var]
        if var == "structural_type":
            g = g.sort_values("d_rps_lightgbm")
        for r in g.itertuples():
            labels.append(str(r.level).replace("_", " ").replace("rc ", "RC ")), ys.append(y), rows.append((y, r))
            y -= 1
        y -= 0.4
    fig, ax = plt.subplots(figsize=(COL_WIDTH_IN, 4.2))
    ax.axvline(0, color=MUTED, lw=1)
    for i, m in ((1, "ordinal_logit"), (2, "lightgbm")):
        ax.scatter([getattr(r, f"d_rps_{m}") for _, r in rows], [yy + (0.15 if i == 1 else -0.15) for yy, _ in rows],
                   s=16, color=CATEGORICAL[i], edgecolor=SURFACE, lw=0.6, label=LABEL[m], zorder=3)
    ax.set_yticks(ys, labels)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab in dict(sections).values():
            t.set_fontweight("bold"), t.set_color(INK)
    ax.set(xlabel="RPS minus fragility curves (negative = better)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", fontsize=6)
    save(fig, C.FIGURES_DIR / f"12_rps_by_subgroup{sfx}.png")    # caption: LODO, preferred map, groups >= 0.5%


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="smoke test: 50k buildings, one 30k/10k leakage fold")
    a = p.parse_args()
    sfx = "_quick" if a.quick else ""
    pd.set_option("display.width", 220)
    t0, meta = time.time(), {"args": vars(a), "python": platform.python_version(),
                             "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    bd, dec, tw = [], [], []
    for src in C.SHAKEMAP_SOURCES:
        b = lodo_frame(src, a.quick)
        bd.append(breakdown(b, src))
        dec.append(decomposition(b, src))
        tw.append(twins(b, src))
    bd, dec, tw = pd.concat(bd), pd.concat(dec), pd.DataFrame(tw)
    bd.to_csv(T / f"12_breakdown{sfx}.csv", index=False)
    dec.to_csv(T / f"12_residual_decomposition{sfx}.csv", index=False)
    tw.to_csv(T / f"12_twins{sfx}.csv", index=False)

    print("A. Error breakdown on new districts (LODO out-of-fold, municipality PGA). d_rps_*: model minus")
    print("fragility curves (negative = better).")
    cols = ["variable", "level", "n", "observed_severe", "pred_severe_fragility", "pred_severe_lightgbm",
            "rps_fragility", "rps_ordinal_logit", "rps_lightgbm", "d_rps_ordinal_logit", "d_rps_lightgbm"]
    for src, g in bd.groupby("pga_source", sort=False):
        print(f"\n{src}:")
        print(g[cols].round(4).to_string(index=False))
    print("\nB. Where the error sits: nested split of the mean squared value (sums to 'total'). Outcome row:")
    print("variance of the severe indicator; residual rows: Brier score of each model on new districts.")
    print(dec.round(4).to_string(index=False))
    print("\nC. Twins (same ward, identical inputs): grade agreement and the RPS noise floor on these buildings.")
    print(tw.T.to_string())

    leak, summary, facts = leakage_demo(a)
    leak.to_csv(T / f"12_leakage_demo_metrics{sfx}.csv", index=False)
    summary.to_csv(T / f"12_leakage_demo_summary{sfx}.csv", index=False)
    print("\nD. LEAKAGE DEMONSTRATION (post-earthquake columns are NOT valid inputs; shown only to quantify the")
    print(f"inflation). Municipality folds, seed {C.SEED}, {C.SHAKEMAP_PRIMARY} map, same folds for all rows.")
    print(summary.round(4).to_string(index=False))
    print(json.dumps(facts, indent=2))
    meta |= {"leakage_facts": facts, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
             "minutes": round((time.time() - t0) / 60, 1)}
    (T / f"12_run{sfx}.json").write_text(json.dumps(meta, indent=2, default=str))
    figure(bd, sfx)


if __name__ == "__main__":
    main()
