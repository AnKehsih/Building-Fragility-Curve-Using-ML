"""
09c_lodo_cv.py

Purpose : Leave-one-district-out CV on the 10 development districts (D13). Municipality folds share
          districts between training and validation ("new municipality in a known district"); the
          final test is a new district. This run matches the final-test scenario on development data
          only, to show how much the municipality folds flatter each model and whether the
          conclusions of 08/09 hold for a new district. One fold per district (10 folds, no seed).
          Models (each map): every 08b-selected model, the 08 LightGBM and all 09 LightGBM variants
          (district PGA included: expected to lose its municipality-fold advantage), the fragility
          curves (municipality and district PGA) and the majority baseline. Selection is NOT changed by
          this run (it is a check of the CV design; selection stays on municipality folds, D9/D13).
          Each task also scores 100k random TRAINING rows, so the train-validation gap (overfitting)
          is reported. Random forest training rows are scored out-of-bag (Step 22; the first run,
          Step 21, used in-bag predictions, which overstate the forest's gap).
Inputs  : data/processed/* (as 08), results/tables/08b_selected.csv
Outputs : results/tables/09c_lodo_metrics_<tag>.csv  one row per task (validation + train_* metrics)
          results/tables/09c_run_<tag>.json
          results/oof/09c_<model>_<config>_<pga>_<level>_<fs>.parquet  (all 10 folds)
          With --report (local, after the run): 09c_summary.csv, 09c_paired_vs_fragility.csv,
          09c_vs_municipality_folds.csv, 09c_by_district.csv, 09c_console_output.txt via redirect
Run     : python code/09c_lodo_cv.py --n_jobs 10 --threads 3            (multi-core machine; 210 fits)
          python code/09c_lodo_cv.py --quick --n_jobs 2 --threads 2    (smoke test)
          .venv/Scripts/python code/09c_lodo_cv.py --report > results/tables/09c_console_output.txt
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
from utils.cvrun import KEY, assemble_oof, run_parallel
from utils.models import GRIDS, LGBM_VARIANTS, config_name

T = C.TABLES_DIR
N_DISTRICTS = 10
PAIR = ["fold"]


def build_tasks(a):
    sel = pd.read_csv(T / "08b_selected.csv")
    by_name = {(m, config_name(p)): p for m, grid in GRIDS.items() for p in grid}
    sel = {(r.pga_source, r.model): by_name[(r.model, r.config)] for r in sel.itertuples()}
    tasks = []
    order = {"random_forest": 0, "lightgbm": 1, "ordinal_logit": 2, "decision_tree": 3, "fragility": 4, "majority": 5}

    def add(model, params, fs, src, level, label):
        cfg = config_name(params) if params else "-"
        for fold in range(1 if a.quick else N_DISTRICTS):
            tasks.append({"variant": label, "model": model, "config": cfg, "params": params, "feature_set": fs,
                          "pga_source": src, "pga_level": level, "scheme": "district", "seed": C.SEED, "fold": fold,
                          "train_score": True, "save_oof": True,
                          "oof_name": f"09c_{model}_{cfg}_{src}_{level}_{fs}".replace("-", "na")
                                      + ("_quick" if a.quick else "")})

    for src in C.SHAKEMAP_SOURCES:
        for model in GRIDS:
            add(model, sel[src, model], "full", src, "municipality", "08_default")
        for label, extra, fs, level in LGBM_VARIANTS:
            add("lightgbm", sel[src, "lightgbm"] | extra, fs, src, level, label)
        for level in ("municipality", "district"):
            add("fragility", {}, "type_pga", src, level, "baseline")
    add("majority", {}, "type_pga", C.SHAKEMAP_PRIMARY, "municipality", "baseline")
    return sorted(tasks, key=lambda t: order[t["model"]])


def run(a):
    out = T / f"09c_lodo_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)
    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "lgbm_variants": LGBM_VARIANTS, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    seconds = run_parallel(tasks, out, a.n_jobs, a.threads, a.quick, extra_keys=("variant",))
    assemble_oof({t["oof_name"] for t in build_tasks(a)}, 1 if a.quick else N_DISTRICTS)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (T / f"09c_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    res = pd.read_csv(out)
    print("\n", res.groupby(["pga_source", "model", "variant", "pga_level"])[["rps", "train_rps"]].mean().round(4).to_string())


def report(tag):
    """Local analysis of a finished run (reads CSVs only)."""
    pd.set_option("display.width", 220)
    sfx = "" if tag == "main" else f"_{tag}"          # quick runs must not overwrite the main report
    r = pd.read_csv(T / f"09c_lodo_metrics_{tag}.csv")
    r8, r9 = pd.read_csv(T / "08_cv_metrics_main.csv"), pd.read_csv(T / "09_cv_metrics_main.csv")
    base = pd.read_csv(T / "07_cv_metrics.csv")
    from utils.data import load_development
    dev = load_development()
    ids = sorted(dev["district_id"].unique())
    names = dict(zip(ids, (C.DISTRICTS[d] for d in ids)))
    r["district"] = r["fold"].map(dict(enumerate(names.values())))
    g = ["pga_source", "model", "variant", "config", "feature_set", "pga_level"]

    summ = r.groupby(g).agg(rps=("rps", "mean"), rps_sd=("rps", "std"), brier_severe=("brier_severe", "mean"),
                            log_loss=("log_loss", "mean"), qwk=("qwk", "mean"), train_rps=("train_rps", "mean"),
                            n_folds=("rps", "size")).reset_index()
    if "train_inbag_rps" in r:      # random forest: train_rps is out-of-bag; in-bag shown for comparison
        summ = summ.merge(r.groupby(g, dropna=False)["train_inbag_rps"].mean().reset_index(), on=g)
    w = r.assign(wr=r.rps * r.n).groupby(g)[["wr", "n"]].sum()
    summ = summ.merge((w.wr / w.n).rename("rps_weighted").reset_index(), on=g)
    summ["gap_val_minus_train"] = summ["rps"] - summ["train_rps"]
    # The gap also contains the shift between districts (even the majority baseline, which cannot overfit,
    # has one). The fragility curves (~5 parameters per type) barely overfit, so the gap in excess of theirs
    # (same map, municipality PGA) is the overfitting part.
    fg = summ[(summ.model == "fragility") & (summ.pga_level == "municipality")].set_index("pga_source")["gap_val_minus_train"]
    summ["excess_gap_vs_fragility"] = summ["gap_val_minus_train"] - summ["pga_source"].map(fg)
    summ = summ.sort_values(["pga_source", "rps"])
    summ.to_csv(T / f"09c_summary{sfx}.csv", index=False)
    print("1. Leave-one-district-out CV, mean over the 10 held-out districts (rps_weighted: weighted by buildings).")
    print("   gap = validation RPS minus RPS on 100k training rows; excess gap = gap minus the fragility curves' gap")
    print("   (district shift removed; positive = overfitting):")
    print(summ.drop(columns=["config"]).round(4).to_string(index=False))

    rows = []
    for (src, model, variant, cfg, fs, level), a in r.groupby(g):
        b = r[(r.model == "fragility") & (r.pga_source == src) & (r.pga_level == "municipality")]
        m = a[PAIR + ["rps"]].merge(b[PAIR + ["rps"]], on=PAIR, suffixes=("", "_frag"))
        d = m.rps - m.rps_frag
        rows.append({"pga_source": src, "model": model, "variant": variant, "pga_level": level,
                     "mean": d.mean(), "sd": d.std(), "min": d.min(), "max": d.max(), "n_districts_better": int((d < 0).sum())})
    pv = pd.DataFrame(rows).sort_values(["pga_source", "mean"])
    pv.to_csv(T / f"09c_paired_vs_fragility{sfx}.csv", index=False)
    print("\n2. Paired RPS difference vs fragility curves (same map, municipality PGA), by held-out district:")
    print(pv.round(4).to_string(index=False))

    # municipality-fold score of the same model (08/09/07), for the LODO-vs-municipality gap
    def muni_rps(row):
        if row.model == "majority":
            return base[(base.model == "majority") & (base.scheme == "municipality")].rps.mean()
        if row.model == "fragility":
            return base[(base.model == f"fragility_{row.pga_source}_{row.pga_level}") & (base.scheme == "municipality")].rps.mean()
        src = r9 if row.variant != "08_default" else r8
        m = src[(src.model == row.model) & (src.config == row.config) & (src.pga_source == row.pga_source)
                & (src.pga_level == row.pga_level) & (src.feature_set == row.feature_set) & (src.scheme == "municipality")]
        return m.rps.mean()
    vm = summ[g + ["rps"]].rename(columns={"rps": "rps_lodo"})
    vm["rps_municipality_folds"] = [muni_rps(x) for x in vm.itertuples()]
    vm["lodo_minus_municipality"] = vm["rps_lodo"] - vm["rps_municipality_folds"]
    vm.to_csv(T / f"09c_vs_municipality_folds{sfx}.csv", index=False)
    print("\n3. LODO vs municipality folds (positive = municipality folds flattered the model):")
    print(vm.drop(columns=["config"]).round(4).to_string(index=False))

    bd = r.pivot_table(index=["pga_source", "model", "variant", "pga_level"], columns="district", values="rps").round(4)
    bd.to_csv(T / f"09c_by_district{sfx}.csv")
    print("\n4. RPS by held-out district:")
    print(bd.to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--tag", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 fold, 30k/10k rows, small forests")
    p.add_argument("--report", action="store_true", help="analyse a finished run (local)")
    a = p.parse_args()
    if a.quick:
        a.tag = f"{a.tag}_quick"
    report(a.tag) if a.report else run(a)


if __name__ == "__main__":
    main()
