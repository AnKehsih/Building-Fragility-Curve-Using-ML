"""
09d_capacity_check.py

Purpose : Post-hoc check (added after Step 21, NOT used for selection): can the overfitting of the random
          forest and of the plain LightGBM on a new district be removed by limiting their capacity?
            random forest  min_samples_leaf 20, 100, 500, 2000 (08 grid was 20 and 100). For a forest the
                           number of trees does not cause overfitting (more trees only average out noise);
                           leaf size does.
            LightGBM       the 08-selected configuration with a FIXED number of rounds (10, 20, 40, 100;
                           no early stopping, all training rows), against early stopping (median 50 / 70
                           rounds on LODO, Step 21). Tests whether fewer rounds fix it.
          Same leave-one-district-out folds as 09c (10 folds, full features, municipality PGA, both maps),
          so every row is paired with the 09c fragility curves fold by fold. Training rows are scored as in
          09c (random forest out-of-bag). Selection stays as in 09b_selected_final.csv; under D12 these
          configurations are also scored on the final test, labelled as post-hoc.
Inputs  : data/processed/* (as 08), results/tables/08b_selected.csv; for --report also 09c_lodo_metrics_main.csv
Outputs : results/tables/09d_metrics_<tag>.csv, 09d_run_<tag>.json
          With --report (local): 09d_summary.csv, 09d_console_output.txt via redirect
Run     : python code/09d_capacity_check.py --n_jobs 10 --threads 3            (multi-core machine; 160 fits)
          python code/09d_capacity_check.py --quick --n_jobs 2 --threads 2    (smoke test)
          .venv/Scripts/python code/09d_capacity_check.py --report > results/tables/09d_console_output.txt
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
from utils.cvrun import KEY, run_parallel
from utils.models import GRIDS, config_name

T = C.TABLES_DIR
N_DISTRICTS = 10
RF_LEAVES = (20, 100, 500, 2000)
LGBM_ROUNDS = (10, 20, 40, 100)


def build_tasks(a):
    sel = pd.read_csv(T / "08b_selected.csv")
    by_name = {(m, config_name(p)): p for m, grid in GRIDS.items() for p in grid}
    sel = {(r.pga_source, r.model): by_name[(r.model, r.config)] for r in sel.itertuples()}
    runs = [("random_forest", {"min_samples_leaf": m}) for m in RF_LEAVES]
    tasks = []
    for src in C.SHAKEMAP_SOURCES:
        for model, params in runs + [("lightgbm", sel[src, "lightgbm"] | {"n_rounds": n}) for n in LGBM_ROUNDS]:
            for fold in range(1 if a.quick else N_DISTRICTS):
                tasks.append({"model": model, "config": config_name(params), "params": params, "feature_set": "full",
                              "pga_source": src, "pga_level": "municipality", "scheme": "district", "seed": C.SEED,
                              "fold": fold, "train_score": True, "save_oof": False, "oof_name": ""})
    return tasks


def run(a):
    out = T / f"09d_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)
    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "rf_leaves": RF_LEAVES, "lgbm_rounds": LGBM_ROUNDS, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    seconds = run_parallel(tasks, out, a.n_jobs, a.threads, a.quick)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (T / f"09d_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    res = pd.read_csv(out)
    print("\n", res.groupby(["pga_source", "model", "config"])[["rps", "train_rps"]].mean().round(4).to_string())


def report(tag):
    """Local analysis of a finished run, paired with the 09c fragility curves and early-stopped LightGBM."""
    pd.set_option("display.width", 220)
    sfx = "" if tag == "main" else f"_{tag}"          # quick runs must not overwrite the main report
    r = pd.read_csv(T / f"09d_metrics_{tag}.csv")
    c = pd.read_csv(T / "09c_lodo_metrics_main.csv")
    lg_es = c[(c.model == "lightgbm") & (c.variant == "08_default")].assign(config=lambda d: d.config + "_early_stop")
    r = pd.concat([r, lg_es[r.columns.intersection(lg_es.columns)]], ignore_index=True)
    frag = c[(c.model == "fragility") & (c.pga_level == "municipality")]
    frag = frag.assign(frag_gap=frag.rps - frag.train_rps)[["pga_source", "fold", "rps", "frag_gap"]]
    m = r.merge(frag, on=["pga_source", "fold"], suffixes=("", "_frag"))
    m["d_vs_frag"] = m.rps - m.rps_frag
    m["excess_gap"] = (m.rps - m.train_rps) - m.frag_gap
    agg = dict(rps=("rps", "mean"), d_vs_fragility=("d_vs_frag", "mean"), d_sd=("d_vs_frag", "std"),
               districts_better=("d_vs_frag", lambda d: int((d < 0).sum())), train_rps=("train_rps", "mean"),
               excess_gap=("excess_gap", "mean"), rounds=("best_iteration", "median"), n_folds=("rps", "size"))
    if "train_inbag_rps" in m:
        agg["train_inbag_rps"] = ("train_inbag_rps", "mean")
    s = m.groupby(["pga_source", "model", "config"]).agg(**agg).reset_index().sort_values(["pga_source", "model", "rps"])
    s.to_csv(T / f"09d_summary{sfx}.csv", index=False)
    print("Capacity check, leave-one-district-out (10 held-out districts). d_vs_fragility: paired RPS minus the")
    print("fragility curves (same map, municipality PGA; negative = better). excess_gap: (validation - training RPS)")
    print("minus the fragility curves' gap. Random forest train_rps is out-of-bag; train_inbag_rps uses all trees.")
    print("'_early_stop' rows: the 09c LightGBM (early stopping on held-out municipalities), for reference.")
    print(s.round(4).to_string(index=False))


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
