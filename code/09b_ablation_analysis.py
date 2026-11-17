"""
09b_ablation_analysis.py

Purpose : Reads the 09 CV results (no fitting); all comparisons paired by scheme/seed/fold.
            1. Ablation: each model with full / type_pga / no_pga features (08 rows for "full"),
               paired vs the fragility curves of the same map and vs the same model with full features.
               Splits the ML gain into "model form" (type_pga vs fragility) and "building
               attributes" (full vs type_pga), and measures what PGA adds (full vs no_pga).
            2. LightGBM diagnosis: each variant vs fragility and vs the 08 LightGBM; random-split
               optimism (seed 678, preferred map).
            3. District-PGA check: if district PGA helps LightGBM because it carries shaking
               information, the two maps (different district PGA values) should give different
               scores, and the fragility curves should also gain from it (07). If it helps because
               it identifies the district (districts are shared between municipality folds, but the
               final-test district is new), the score is map-independent and fragility gains nothing.
            4. LightGBM variant ranking by the D9 rule (mean municipality-fold RPS per map);
            5. final selection per model and map (D9 rule; LightGBM among the 08 default and the 09
               variants, district PGA excluded by D13), read by 09c and 13.
Inputs  : results/tables/09_cv_metrics_main.csv, 08_cv_metrics_main.csv, 08b_selected.csv, 07_cv_metrics.csv
Outputs : results/tables/09b_ablation.csv, 09b_lgbm_diagnosis.csv, 09b_district_check.csv,
          09b_lgbm_variant_ranking.csv, 09b_selected_final.csv
Run     : .venv/Scripts/python code/09b_ablation_analysis.py > results/tables/09b_console_output.txt
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.models import EXCLUDED_FROM_SELECTION, GRIDS, LGBM_VARIANTS, config_name

T = C.TABLES_DIR
PAIR = ["scheme", "seed", "fold"]
REF = C.SHAKEMAP_PRIMARY
pd.set_option("display.width", 220)


def diff(a, b):
    """Paired RPS difference a - b: mean, sd, folds where a is better."""
    m = a[PAIR + ["rps"]].merge(b[PAIR + ["rps"]], on=PAIR, suffixes=("_a", "_b"))
    d = m["rps_a"] - m["rps_b"]
    return d.mean(), d.std(), int((d < 0).sum()), len(d)


def main():
    r9 = pd.read_csv(T / "09_cv_metrics_main.csv")
    r8 = pd.read_csv(T / "08_cv_metrics_main.csv")
    base = pd.read_csv(T / "07_cv_metrics.csv")
    sel = pd.read_csv(T / "08b_selected.csv").set_index(["pga_source", "model"])["config"]
    frag = {s: base[base.model == f"fragility_{s}_municipality"] for s in C.SHAKEMAP_SOURCES}
    muni = lambda d: d[d.scheme == "municipality"]

    def full08(src, model, scheme="municipality"):
        return r8[(r8.model == model) & (r8.config == sel[src, model]) & (r8.pga_source == src) & (r8.scheme == scheme)]

    # 1. ablation
    rows = []
    for model in sorted(r8.model.unique()):
        for src in C.SHAKEMAP_SOURCES:
            full = full08(src, model)
            for fs in ("full", "type_pga", "no_pga"):
                if fs == "full":
                    a = full
                else:
                    a = r9[(r9.study == "ablation") & (r9.model == model) & (r9.feature_set == fs)
                           & (r9.pga_source == (REF if fs == "no_pga" else src))]
                d_f, s_f, b_f, n = diff(a, muni(frag[src]))
                d_full = diff(a, full)
                rows.append({"model": model, "pga_source": "none" if fs == "no_pga" else src, "feature_set": fs,
                             "config": sel[REF if fs == "no_pga" else src, model], "n_folds": n,
                             "rps": a.rps.mean(), "rps_sd": a.rps.std(), "brier_severe": a.brier_severe.mean(),
                             "vs_fragility_mean": d_f, "vs_fragility_sd": s_f, "vs_fragility_folds_better": b_f,
                             "fragility_map": src, "vs_full_mean": d_full[0], "vs_full_sd": d_full[1]})
    ab = pd.DataFrame(rows).drop_duplicates(["model", "pga_source", "feature_set", "fragility_map"])
    ab.to_csv(T / "09b_ablation.csv", index=False)
    print("1. Ablation (municipality folds, 15 folds). vs_fragility: paired RPS minus fragility of the same map "
          "(no_pga compared with both maps' fragility); vs_full: minus the same model with full features.")
    print(ab.drop(columns=["config", "n_folds"]).round(4).to_string(index=False))

    # 2. LightGBM diagnosis
    rows = []
    lg = r9[r9.study == "lgbm_diagnosis"]
    for (src, level, fs, cfg), g in lg.groupby(["pga_source", "pga_level", "feature_set", "config"]):
        variant = "district_pga" if level == "district" else cfg.replace(sel[src, "lightgbm"], "").strip("_") or "?"
        m = muni(g)
        row = {"variant": variant, "pga_source": src, "config": cfg, "rps": m.rps.mean(), "rps_sd": m.rps.std(),
               "brier_severe": m.brier_severe.mean(), "median_rounds": m.best_iteration.median()}
        row["vs_fragility_mean"], row["vs_fragility_sd"], row["vs_fragility_folds_better"], _ = diff(m, muni(frag[src]))
        row["vs_lgbm08_mean"], row["vs_lgbm08_sd"], row["vs_lgbm08_folds_better"], _ = diff(m, full08(src, "lightgbm"))
        rnd = g[g.scheme == "random"]
        if len(rnd):
            mm = m[m.seed == C.SEED]
            row["rps_random_s678"], row["random_optimism_pct"] = rnd.rps.mean(), 100 * (mm.rps.mean() - rnd.rps.mean()) / mm.rps.mean()
        rows.append(row)
    for src in C.SHAKEMAP_SOURCES:                                      # 08 LightGBM for reference
        m, rnd = full08(src, "lightgbm"), full08(src, "lightgbm", "random")
        row = {"variant": "08_default", "pga_source": src, "config": sel[src, "lightgbm"], "rps": m.rps.mean(),
               "rps_sd": m.rps.std(), "brier_severe": m.brier_severe.mean(), "median_rounds": m.best_iteration.median()}
        row["vs_fragility_mean"], row["vs_fragility_sd"], row["vs_fragility_folds_better"], _ = diff(m, muni(frag[src]))
        if len(rnd):
            mm = m[m.seed == C.SEED]
            row["rps_random_s678"], row["random_optimism_pct"] = rnd.rps.mean(), 100 * (mm.rps.mean() - rnd.rps.mean()) / mm.rps.mean()
        rows.append(row)
    dg = pd.DataFrame(rows).sort_values(["pga_source", "rps"])
    dg.to_csv(T / "09b_lgbm_diagnosis.csv", index=False)
    print("\n2. LightGBM variants (municipality folds, 15 folds); random optimism = seed 678, preferred map:")
    print(dg.drop(columns=["config"]).round(4).to_string(index=False))

    # 3. district-PGA check
    dist = muni(lg[lg.pga_level == "district"])
    a, b = (dist[dist.pga_source == s] for s in C.SHAKEMAP_SOURCES)
    rows = [{"check": "LightGBM district PGA: neic_v9 - preferred", **dict(zip(["mean", "sd", "folds_better", "n"], diff(b, a)))}]
    for s in C.SHAKEMAP_SOURCES:
        fd, fm = base[(base.model == f"fragility_{s}_district")], frag[s]
        rows.append({"check": f"fragility district - municipality PGA ({s})",
                     **dict(zip(["mean", "sd", "folds_better", "n"], diff(muni(fd), muni(fm))))})
        rows.append({"check": f"LightGBM district - municipality PGA ({s})",
                     **dict(zip(["mean", "sd", "folds_better", "n"], diff(dist[dist.pga_source == s], full08(s, "lightgbm"))))})
    dc = pd.DataFrame(rows)
    dc.to_csv(T / "09b_district_check.csv", index=False)
    print("\n3. District-PGA check (paired RPS differences, municipality folds):")
    print(dc.round(4).to_string(index=False))

    # 4. LightGBM variant ranking (D9 rule)
    rk = dg[["variant", "pga_source", "config", "rps", "rps_sd"]].sort_values(["pga_source", "rps"])
    rk["rank"] = rk.groupby("pga_source")["rps"].rank().astype(int)
    rk.to_csv(T / "09b_lgbm_variant_ranking.csv", index=False)
    print("\n4. LightGBM variants ranked by mean municipality-fold RPS (D9 rule), per map:")
    print(rk.round(4).to_string(index=False))

    # 5. final selection per model and map (D9 rule; LightGBM chosen among 08 default + 09 variants,
    #    excluding EXCLUDED_FROM_SELECTION, D13). This is the list marked as "selected" in the final test.
    params08 = {(m, config_name(p)): p for m, grid in GRIDS.items() for p in grid}
    rows = []
    for src in C.SHAKEMAP_SOURCES:
        for model in sorted(r8.model.unique()):
            p = params08[model, sel[src, model]]
            cands = [("08_default", p, "full", "municipality", full08(src, model).rps.mean())]
            if model == "lightgbm":
                for label, extra, fs, level in LGBM_VARIANTS:
                    if label in EXCLUDED_FROM_SELECTION:
                        continue
                    g = muni(lg[(lg.pga_source == src) & (lg.pga_level == level) & (lg.feature_set == fs)
                                & (lg.config == config_name(p | extra))])
                    cands.append((label, p | extra, fs, level, g.rps.mean()))
            label, params, fs, level, rps = min(cands, key=lambda c: c[4])
            rows.append({"pga_source": src, "model": model, "variant": label, "config": config_name(params),
                         "feature_set": fs, "pga_level": level, "params": json.dumps(params), "cv_rps": rps})
    fin = pd.DataFrame(rows)
    fin.to_csv(T / "09b_selected_final.csv", index=False)
    print(f"\n5. Final selection per model and map (excluded from selection: {EXCLUDED_FROM_SELECTION}):")
    print(fin.drop(columns=["params"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
