"""
08b_model_selection.py

Purpose : Reads the 08 CV results (no fitting) and
            1. summarises every model x config x PGA source x scheme (mean, SD over folds);
            2. selects ONE configuration per model and PGA map (D11: both maps reported side by side):
               lowest mean RPS on municipality folds with that map (pre-registered rule D9,
               CONTEXT 5.2). The final-test district is not used;
            3. selection optimism per map: best vs median configuration of each grid (how much
               picking the best of a grid flatters the CV score);
            4. paired fold differences vs the fragility baseline of 07 (same PGA map, scheme,
               seed, fold), negative = ML better;
            5. map effect: NEIC v9 vs preferred PGA, paired by fold, each model at its configuration
               selected on each map (the D7 check of Step 11: gain vs fold-to-fold SD of RPS);
            6. random vs municipality folds (seed 678, reference map; random folds were only run with
               it): how much a random split flatters each model;
            7. LightGBM early-stopping rounds per scheme.
Inputs  : results/tables/08_cv_metrics_main.csv, results/tables/07_cv_metrics.csv
Outputs : results/tables/08b_summary.csv, 08b_selected.csv, 08b_selection_optimism.csv,
          08b_paired_vs_fragility.csv, 08b_map_check.csv, 08b_random_vs_municipality.csv,
          08b_lgbm_rounds.csv
Run     : .venv/Scripts/python code/08b_model_selection.py > results/tables/08b_console_output.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

T = C.TABLES_DIR
METRICS = ["rps", "brier_severe", "log_loss", "qwk", "macro_f1", "accuracy"]
PAIR = ["scheme", "seed", "fold"]
REF, OTHER = C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY
pd.set_option("display.width", 200)


def paired(a, b, label):
    """RPS of a minus RPS of b, matched on scheme/seed/fold."""
    m = a[PAIR + ["rps"]].merge(b[PAIR + ["rps"]], on=PAIR, suffixes=("_a", "_b"))
    d = m["rps_a"] - m["rps_b"]
    return {"comparison": label, "n_folds": len(d), "mean": d.mean(), "sd": d.std(), "min": d.min(),
            "max": d.max(), "n_folds_better": int((d < 0).sum()),
            "rps_a": m["rps_a"].mean(), "rps_b": m["rps_b"].mean()}


def main():
    r = pd.read_csv(T / "08_cv_metrics_main.csv")
    base = pd.read_csv(T / "07_cv_metrics.csv")
    assert len(r) == r.drop_duplicates(["model", "config", "pga_source", "scheme", "seed", "fold"]).shape[0]
    frag = {src: base[base.model == f"fragility_{src}_municipality"] for src in C.SHAKEMAP_SOURCES}
    majority = base[base.model == "majority"]

    def runs(model, config, src, scheme):
        return r[(r.model == model) & (r.config == config) & (r.pga_source == src) & (r.scheme == scheme)]

    # 1. summary
    g = ["scheme", "pga_source", "model", "config"]
    summ = r.groupby(g)[METRICS].agg(["mean", "std"])
    summ.columns = [f"{m}_{s}" for m, s in summ.columns]
    summ["n_folds"] = r.groupby(g).size()
    summ = summ.reset_index()
    summ.to_csv(T / "08b_summary.csv", index=False)

    # 2. selection per map (municipality folds)
    sel_pool = summ[summ.scheme == "municipality"]
    selected = (sel_pool.loc[sel_pool.groupby(["pga_source", "model"])["rps_mean"].idxmin()]
                .sort_values(["pga_source", "rps_mean"]))
    selected.to_csv(T / "08b_selected.csv", index=False)
    cfg = selected.set_index(["pga_source", "model"])["config"]
    print("Baselines (07, municipality folds, 15 folds):")
    for name in ["majority"] + [f"fragility_{src}_municipality" for src in C.SHAKEMAP_SOURCES]:
        b = base[(base.model == name) & (base.scheme == "municipality")]
        print(f"  {name:35s} RPS {b.rps.mean():.4f} ± {b.rps.std():.4f}  Brier>=4 {b.brier_severe.mean():.4f}  "
              f"log loss {b.log_loss.mean():.4f}  QWK {b.qwk.mean():.4f}")
    print("\nSelected configuration per model and map (municipality folds, 15 folds):")
    print(selected[["pga_source", "model", "config", "rps_mean", "rps_std", "brier_severe_mean", "log_loss_mean",
                    "qwk_mean", "macro_f1_mean", "accuracy_mean"]].round(4).to_string(index=False))

    # 3. selection optimism
    opt = (sel_pool.groupby(["pga_source", "model"])["rps_mean"]
           .agg(best="min", median="median", worst="max", n_configs="size").reset_index())
    opt["best_minus_median"] = opt["best"] - opt["median"]
    opt.to_csv(T / "08b_selection_optimism.csv", index=False)
    print("\nSelection optimism (mean RPS over configs of each grid):")
    print(opt.round(4).to_string(index=False))

    # 4. paired differences vs fragility (same map) and majority
    rows = []
    for (src, model), config in cfg.items():
        for scheme in ("municipality", "random"):
            a = runs(model, config, src, scheme)
            if len(a):
                rows.append({"model": model, "config": config, "pga_source": src, "scheme": scheme}
                            | paired(a, frag[src][frag[src].scheme == scheme], f"{model} - fragility ({src})"))
        rows.append({"model": model, "config": config, "pga_source": src, "scheme": "municipality"}
                    | paired(runs(model, config, src, "municipality"), majority[majority.scheme == "municipality"],
                             f"{model} ({src}) - majority"))
    pv = pd.DataFrame(rows)
    pv.to_csv(T / "08b_paired_vs_fragility.csv", index=False)
    print("\nPaired RPS differences, selected config minus baseline (negative = ML better):")
    print(pv[["comparison", "scheme", "n_folds", "mean", "sd", "min", "max", "n_folds_better"]]
          .round(4).to_string(index=False))

    # 5. map effect, each model at its configuration selected on each map
    rows = []
    for model in sorted(r.model.unique()):
        a, b = runs(model, cfg[OTHER, model], OTHER, "municipality"), runs(model, cfg[REF, model], REF, "municipality")
        rows.append({"model": model, "config_" + OTHER: cfg[OTHER, model], "config_" + REF: cfg[REF, model]}
                    | paired(a, b, f"{model}: {OTHER} - {REF}") | {"rps_sd_folds_ref": b.rps.std()})
    a, b = (frag[s][frag[s].scheme == "municipality"] for s in (OTHER, REF))
    rows.append({"model": "fragility"} | paired(a, b, f"fragility: {OTHER} - {REF}") | {"rps_sd_folds_ref": b.rps.std()})
    mc = pd.DataFrame(rows)
    mc["gain_exceeds_fold_sd"] = -mc["mean"] > mc["rps_sd_folds_ref"]
    mc.to_csv(T / "08b_map_check.csv", index=False)
    print(f"\nMap effect, {OTHER} minus {REF} (negative = {OTHER} better), vs fold-to-fold SD of RPS:")
    print(mc[["comparison", "mean", "sd", "n_folds_better", "rps_sd_folds_ref", "gain_exceeds_fold_sd"]]
          .round(4).to_string(index=False))

    # 6. random vs municipality (seed 678, reference map: random folds were run for that seed and map only)
    rows = []
    for model in sorted(r.model.unique()):
        m = r[(r.model == model) & (r.config == cfg[REF, model]) & (r.pga_source == REF) & (r.seed == C.SEED)]
        rows.append({"model": model, "config": cfg[REF, model],
                     "rps_municipality": m[m.scheme == "municipality"].rps.mean(),
                     "rps_random": m[m.scheme == "random"].rps.mean()})
    for name in [f"fragility_{REF}_municipality", "majority"]:
        m = base[(base.model == name) & (base.seed == C.SEED)]
        rows.append({"model": name, "config": "-", "rps_municipality": m[m.scheme == "municipality"].rps.mean(),
                     "rps_random": m[m.scheme == "random"].rps.mean()})
    rv = pd.DataFrame(rows)
    rv["optimism"] = rv["rps_municipality"] - rv["rps_random"]
    rv["optimism_pct"] = 100 * rv["optimism"] / rv["rps_municipality"]
    rv.to_csv(T / "08b_random_vs_municipality.csv", index=False)
    print(f"\nRandom vs municipality folds (seed {C.SEED}, {REF} map); optimism = how much a random split flatters:")
    print(rv.round(4).to_string(index=False))

    # 7. LightGBM rounds
    lg = r[r.model == "lightgbm"]
    rounds = lg.groupby(["scheme", "pga_source", "config"])["best_iteration"].describe()[["min", "50%", "max"]]
    rounds.to_csv(T / "08b_lgbm_rounds.csv")
    print("\nLightGBM early-stopping rounds (of 3000):")
    print(rounds.to_string())


if __name__ == "__main__":
    main()
