"""
10_calibration_sensitivity.py

Purpose : Graduate analysis 2 (CONTEXT 6): calibration of severe-damage probabilities and sensitivity to the
          resolution of shaking (municipality vs district PGA). Descriptive: changes neither D6 (municipality
          PGA) nor the 09b selection, and nothing is recalibrated.
          A. Fits (the only new ones): the two selected models (ordinal logit; LightGBM boosted from the
             fragility curve; 09b_selected_final.csv) with DISTRICT PGA, both maps, on municipality folds
             (5 x 3 seeds, paired with 08 / 09) and leave-one-district-out (10 folds, paired with 09c;
             out-of-fold predictions saved). Both models are smooth in PGA (linear in ln PGA, or through the
             fragility curve), so district PGA cannot act as a district identifier the way it does for the
             plain LightGBM (D13); LODO is still the valid scheme for a new district.
          B. Resolution sensitivity table: RPS and severe Brier with municipality vs district PGA, paired
             fold by fold, for the fragility curves (07 / 09c), the two selected models and the plain
             LightGBM (08 / 09 / 09c, reference: the D13 case).
          C. Calibration of P(grade >= 4) on new districts (LODO out-of-fold predictions, all development
             buildings, each predicted by the model that did not see its district): reliability curve
             (10 equal-width bins), Brier decomposition (reliability - resolution + uncertainty), expected
             calibration error, calibration slope (logistic regression of the outcome on logit p), mean
             predicted vs observed share per district and per municipality. Uncertainty: cluster bootstrap
             over municipalities (2,000 resamples) of the paired RPS / Brier difference vs the fragility
             curves.
Inputs  : data/processed/* (as 08), results/tables/09b_selected_final.csv, 08b_selected.csv; for --report also
          07_cv_metrics.csv, 08_cv_metrics_main.csv, 09_cv_metrics_main.csv, 09c_lodo_metrics_main.csv and the
          09c out-of-fold predictions (results/oof/09c_*.parquet)
Outputs : results/tables/10_metrics_<tag>.csv, 10_run_<tag>.json; results/oof/10_*.parquet
          With --report: 10_resolution_sensitivity.csv, 10_calibration.csv, 10_calibration_by_district.csv,
          10_bootstrap_vs_fragility.csv, 10_reliability_bins.csv, 10_console_output.txt via redirect;
          results/figures/10_reliability.png, 10_district_calibration.png
Run     : python code/10_calibration_sensitivity.py --n_jobs 8 --threads 3            (multi-core machine; 100 fits)
          python code/10_calibration_sensitivity.py --quick --n_jobs 2 --threads 2   (smoke test)
          python code/10_calibration_sensitivity.py --report > results/tables/10_console_output.txt
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
from utils.cvrun import KEY, OOF_DIR, assemble_oof, run_parallel
from utils.metrics import p_severe, rps

T = C.TABLES_DIR
N_DISTRICTS = 10
MODELS = ("ordinal_logit", "lightgbm")
LABEL = {"fragility": "Fragility curves", "ordinal_logit": "Ordinal logit", "lightgbm": "Hybrid LightGBM",
         "lightgbm_plain": "Plain LightGBM"}
LEVELS = ("municipality", "district")
N_BINS, N_BOOT, MIN_BIN = 10, 2000, 100     # reliability bins; bootstrap resamples; buildings to plot a bin


def selected():
    """{(map, model): (params, config, feature_set)} for the two models, from 09b."""
    s = pd.read_csv(T / "09b_selected_final.csv")
    return {(r.pga_source, r.model): (json.loads(r.params), r.config, r.feature_set)
            for r in s.itertuples() if r.model in MODELS}


def oof_name(model, cfg, src, fs, quick=False):
    return f"10_{model}_{cfg}_{src}_district_{fs}" + ("_quick" if quick else "")


def build_tasks(a):
    sel, tasks = selected(), []
    seeds = C.CV_REPEAT_SEEDS[:1] if a.quick else C.CV_REPEAT_SEEDS
    n_muni, n_lodo = (1, 1) if a.quick else (C.N_FOLDS, N_DISTRICTS)
    for src in C.SHAKEMAP_SOURCES:
        for model in MODELS:
            params, cfg, fs = sel[src, model]
            base = {"model": model, "config": cfg, "params": params, "feature_set": fs, "pga_source": src,
                    "pga_level": "district", "train_score": False}
            tasks += [base | {"scheme": "municipality", "seed": s, "fold": f, "save_oof": False, "oof_name": ""}
                      for s in seeds for f in range(n_muni)]
            tasks += [base | {"scheme": "district", "seed": C.SEED, "fold": f, "save_oof": True,
                              "oof_name": oof_name(model, cfg, src, fs, a.quick)} for f in range(n_lodo)]
    return tasks


def run(a):
    out = T / f"10_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)
    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    seconds = run_parallel(tasks, out, a.n_jobs, a.threads, a.quick)
    assemble_oof({t["oof_name"] for t in build_tasks(a) if t["save_oof"]}, 1 if a.quick else N_DISTRICTS)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (T / f"10_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------- B. resolution sensitivity
def resolution_scores(r10):
    """Fold scores (seed, fold, rps, brier_severe) per model, map, PGA level and scheme."""
    sel = selected()
    s8 = pd.read_csv(T / "08b_selected.csv")
    m89 = pd.concat([pd.read_csv(T / "08_cv_metrics_main.csv"), pd.read_csv(T / "09_cv_metrics_main.csv")])
    c9, b7 = pd.read_csv(T / "09c_lodo_metrics_main.csv"), pd.read_csv(T / "07_cv_metrics.csv")
    out = []

    def take(d, label, src, level, scheme):
        d = d[(d.pga_source == src) & (d.pga_level == level) & (d.scheme == scheme)].drop_duplicates(["seed", "fold"])
        out.append(d[["seed", "fold", "rps", "brier_severe"]].assign(model=label, pga_source=src, pga_level=level,
                                                                      scheme=scheme))

    for src in C.SHAKEMAP_SOURCES:
        for level in LEVELS:
            take(b7[b7.model == f"fragility_{src}_{level}"].assign(pga_source=src, pga_level=level),
                 "fragility", src, level, "municipality")
            take(c9[c9.model == "fragility"], "fragility", src, level, "district")
        for model in MODELS:
            _, cfg, fs = sel[src, model]
            pick = lambda d: d[(d.model == model) & (d.config == cfg) & (d.feature_set == fs)]
            for scheme, full in (("municipality", m89), ("district", c9)):
                take(pick(full), model, src, "municipality", scheme)
                take(pick(r10), model, src, "district", scheme)
        cfg8 = s8[(s8.scheme == "municipality") & (s8.pga_source == src) & (s8.model == "lightgbm")].config.iloc[0]
        pick = lambda d: d[(d.model == "lightgbm") & (d.config == cfg8) & (d.feature_set == "full")]
        for scheme, full in (("municipality", m89), ("district", c9)):
            for level in LEVELS:
                take(pick(full), "lightgbm_plain", src, level, scheme)
    return pd.concat(out, ignore_index=True)


def resolution_table(sc):
    rows = []
    for (src, model, scheme), g in sc.groupby(["pga_source", "model", "scheme"], sort=False):
        w = g.pivot_table(index=["seed", "fold"], columns="pga_level", values=["rps", "brier_severe"]).dropna()
        if w.empty:
            continue
        d = w["rps"]["district"] - w["rps"]["municipality"]
        rows.append({"pga_source": src, "model": model, "scheme": scheme, "n_folds": len(d),
                     "rps_municipality_pga": w["rps"]["municipality"].mean(), "rps_district_pga": w["rps"]["district"].mean(),
                     "d_district_minus_muni": d.mean(), "d_se": d.std() / np.sqrt(len(d)),
                     "folds_muni_better": int((d > 0).sum()),
                     "d_brier_severe": (w["brier_severe"]["district"] - w["brier_severe"]["municipality"]).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- C. calibration on new districts
def lodo_predictions(tag):
    """{(map, model, level): DataFrame building_id, p_severe, rps} from the LODO out-of-fold files."""
    sel, q = selected(), "_quick" if tag.endswith("quick") else ""
    files = {}
    for src in C.SHAKEMAP_SOURCES:
        for level in LEVELS:
            files[src, "fragility", level] = f"09c_fragility_na_{src}_{level}_type_pga"
        for model in MODELS:
            _, cfg, fs = sel[src, model]
            files[src, model, "municipality"] = f"09c_{model}_{cfg}_{src}_municipality_{fs}"
            files[src, model, "district"] = oof_name(model, cfg, src, fs) + q
    return files


def calibration_stats(p, y):
    """Severe-damage calibration of probabilities p for outcomes y (0/1)."""
    from sklearn.linear_model import LogisticRegression
    b = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    n_k = np.bincount(b, minlength=N_BINS)
    p_k = np.bincount(b, p, N_BINS) / np.maximum(n_k, 1)
    o_k = np.bincount(b, y, N_BINS) / np.maximum(n_k, 1)
    w, obar = n_k / len(p), y.mean()
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    slope = LogisticRegression(C=np.inf).fit(lp[:, None], y).coef_[0, 0]          # unpenalised
    stats = {"n": len(p), "mean_p": p.mean(), "observed": obar, "brier": np.mean((p - y) ** 2),
             "reliability": np.sum(w * (p_k - o_k) ** 2), "resolution": np.sum(w * (o_k - obar) ** 2),
             "uncertainty": obar * (1 - obar), "ece": np.sum(w * np.abs(p_k - o_k)), "slope": slope}
    bins = pd.DataFrame({"bin": range(N_BINS), "n": n_k, "mean_p": p_k, "observed": o_k})
    return stats, bins


def bootstrap_ci(d, groups, n, seed=C.SEED):
    """Cluster bootstrap over municipalities of the mean of per-building values d."""
    s = pd.DataFrame({"g": groups, "d": d}).groupby("g")["d"].agg(["sum", "size"])
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    boot = s["sum"].to_numpy()[idx].sum(1) / s["size"].to_numpy()[idx].sum(1)
    return d.mean(), *np.percentile(boot, [2.5, 97.5])


def figures(bins, by_district, sfx):
    import matplotlib.pyplot as plt
    from utils.plotting import CATEGORICAL, FULL_WIDTH_IN, INK, MUTED, SURFACE, save, set_style
    set_style()
    order = ["fragility", "ordinal_logit", "lightgbm"]                 # fixed colour slots
    titles = {"preferred": "Preferred ShakeMap", "neic_v9": "NEIC v9 ShakeMap"}

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.9), sharey=True)
    for ax, src in zip(axes, C.SHAKEMAP_SOURCES):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED)
        for i, m in enumerate(order):
            b = bins[(bins.pga_source == src) & (bins.model == m) & (bins.pga_level == "municipality") & (bins.n >= MIN_BIN)]
            ax.plot(b.mean_p, b.observed, marker="o", ms=4, color=CATEGORICAL[i], label=LABEL[m])
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted P(grade ≥ 4)", title=titles[src], aspect="equal")
    axes[0].set_ylabel("Observed share with grade ≥ 4")
    axes[0].legend(loc="upper left")
    save(fig, C.FIGURES_DIR / f"10_reliability{sfx}.png")           # caption: LODO, municipality PGA, bins ≥ 100

    # dot plot: one row per held-out district (sorted by observed share), observed = ink diamond
    d = by_district[by_district.pga_level == "municipality"]
    rows = d.groupby("district")["observed"].first().sort_values().index
    y = {name: k for k, name in enumerate(rows)}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.0), sharey=True)
    for ax, src in zip(axes, C.SHAKEMAP_SOURCES):
        s = d[d.pga_source == src]
        for i, m in enumerate(order):
            g = s[s.model == m]
            ax.scatter(g.mean_p, g.district.map(y) + (i - 1) * 0.18, s=20, color=CATEGORICAL[i],
                       edgecolor=SURFACE, lw=0.8, label=LABEL[m], zorder=3)
        o = s.drop_duplicates("district")
        ax.scatter(o.observed, o.district.map(y), marker="D", s=22, color=INK, label="Observed", zorder=4)
        ax.set(xlim=(0, 1), xlabel="Share with grade ≥ 4", title=titles[src])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), rows)
    axes[1].legend(loc="lower right")
    save(fig, C.FIGURES_DIR / f"10_district_calibration{sfx}.png")  # caption: each district predicted by LODO


def report(tag):
    pd.set_option("display.width", 220)
    sfx = "" if tag == "main" else f"_{tag}"          # quick runs must not overwrite the main report
    from utils.data import load_development

    # B. resolution sensitivity
    res = resolution_table(resolution_scores(pd.read_csv(T / f"10_metrics_{tag}.csv")))
    res.to_csv(T / f"10_resolution_sensitivity{sfx}.csv", index=False)
    print("B. Shaking resolution: district minus municipality PGA, paired fold by fold (positive = municipality")
    print("PGA better). scheme municipality: 5 folds x 3 seeds; district: 10 held-out districts. Plain LightGBM: D13 case.")
    print(res.round(5).to_string(index=False))

    # C. calibration on new districts
    dev = load_development()[["building_id", C.TARGET, C.GROUP_COL, "district_id"]]
    cal, bins, by_d, boot, ref = [], [], [], [], {}
    for (src, model, level), name in lodo_predictions(tag).items():
        path = OOF_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"  missing {path.name}; skipped")
            continue
        o = pd.read_parquet(path).merge(dev, on="building_id")
        P = o[[f"p{g}" for g in C.GRADES]].to_numpy(float)
        y = o[C.TARGET].to_numpy(int)
        ps, ys = p_severe(P), (y >= C.SEVERE_MIN_GRADE).astype(float)
        st, b = calibration_stats(ps, ys)
        o = o.assign(ps=ps, ys=ys, rps=rps(P, y), bs=(ps - ys) ** 2)
        mun = o.groupby(C.GROUP_COL)[["ps", "ys"]].mean()
        key = {"pga_source": src, "model": model, "pga_level": level}
        cal.append(key | st | {"rps": o.rps.mean(), "muni_mae": (mun.ps - mun.ys).abs().mean(),
                               "muni_corr": mun.ps.corr(mun.ys)})
        bins.append(b.assign(**key))
        d = o.groupby("district_id")[["ps", "ys"]].mean().rename(columns={"ps": "mean_p", "ys": "observed"})
        by_d.append(d.reset_index().assign(district=lambda x: x.district_id.map(C.DISTRICTS), **key))
        ref[src, model, level] = o.set_index("building_id")[["rps", "bs", C.GROUP_COL]]
    cal, bins, by_d = pd.DataFrame(cal), pd.concat(bins), pd.concat(by_d)
    for (src, model, level), o in ref.items():
        if model == "fragility" or (src, "fragility", level) not in ref:
            continue
        f = ref[src, "fragility", level].loc[o.index]
        for metric, col in (("rps", "rps"), ("brier_severe", "bs")):
            m, lo, hi = bootstrap_ci((o[col] - f[col]).to_numpy(), o[C.GROUP_COL].to_numpy(), N_BOOT)
            boot.append({"pga_source": src, "model": model, "pga_level": level, "metric": metric,
                         "d_vs_fragility": m, "ci_low": lo, "ci_high": hi})
    boot = pd.DataFrame(boot)
    cal.to_csv(T / f"10_calibration{sfx}.csv", index=False)
    bins.to_csv(T / f"10_reliability_bins{sfx}.csv", index=False)
    by_d.to_csv(T / f"10_calibration_by_district{sfx}.csv", index=False)
    boot.to_csv(T / f"10_bootstrap_vs_fragility{sfx}.csv", index=False)
    print("\nC. Calibration of P(grade >= 4) on new districts (LODO out-of-fold, all development buildings).")
    print("brier = reliability - resolution + uncertainty (binned, approximate); ece: expected calibration error;")
    print("slope: 1 = well spread, < 1 = too extreme, > 1 = too timid; muni_*: mean predicted vs observed share per")
    print("municipality (110).")
    print(cal.round(4).to_string(index=False))
    print("\nPer district (municipality PGA): mean predicted vs observed share with grade >= 4")
    w = by_d[by_d.pga_level == "municipality"].pivot_table(index=["pga_source", "district"], columns="model",
                                                           values="mean_p")
    w["observed"] = by_d[by_d.pga_level == "municipality"].groupby(["pga_source", "district"])["observed"].first()
    print(w.round(3).to_string())
    print(f"\nPaired difference vs the fragility curves (same map and PGA level), with 95% cluster-bootstrap")
    print(f"intervals over municipalities ({N_BOOT} resamples). Negative = better than the fragility curves.")
    print(boot.round(5).to_string(index=False))
    figures(bins, by_d, sfx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--tag", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 fold per scheme, 30k/10k rows")
    p.add_argument("--report", action="store_true", help="analyse a finished run")
    a = p.parse_args()
    if a.quick:
        a.tag = f"{a.tag}_quick"
    report(a.tag) if a.report else run(a)


if __name__ == "__main__":
    main()
