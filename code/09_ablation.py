"""
09_ablation.py

Purpose : Two sets of CV runs on the development districts (same folds and metrics as 07/08, so
          every result is paired fold by fold with the fragility baseline and with 08).
          A. Feature ablation (CONTEXT 5.4, graduate analysis): each model at its configuration
             selected in 08b (per map, D11), with
               type_pga  structural type + ln PGA only: the fragility curve's information, so any
                         gain over fragility comes from the model form, not from building attributes;
               no_pga    building attributes + structural type, no PGA: what the ShakeMap adds.
                         (No PGA input, so the map does not matter; run once, with the preferred map's
                         configurations.)
          B. LightGBM diagnosis (Step 16: LightGBM is worse than fragility on municipality folds but
             best on random folds; early stopping after ~30 rounds). Hypothesis: PGA is constant
             within a municipality, so fine splits on ln PGA identify municipalities and the trees
             learn municipality-specific damage rates that do not transfer. Variants of the
             selected LightGBM configuration, each a pre-declared test of that hypothesis:
               district PGA     PGA averaged by district: cannot identify a municipality;
               pga_bins 5, 10   ln PGA limited to 5 or 10 histogram bins (by building counts);
               init fragility   boost from the fragility curve (type + PGA, fitted on the
                                inner-training rows); trees see building attributes but not PGA, so
                                PGA enters only through the smooth, monotone fragility curve.
             All on municipality folds (both maps, 3 seeds) plus random folds (preferred map,
             seed 678) to measure each variant's random-split optimism.
          A variant from B can replace the 08 LightGBM configuration only by the D9 rule (lower
          mean municipality-fold RPS); any such change is logged as a decision before the final test.
Inputs  : data/processed/* (as 08), results/tables/08b_selected.csv
Outputs : results/tables/09_cv_metrics_<tag>.csv   one row per task (column `study`: ablation / lgbm_diagnosis)
          results/tables/09_run_<tag>.json
          results/oof/09_<model>_<config>_<pga>_<level>_<fs>_s<seed>.parquet  (municipality folds, first seed)
Run     : python code/09_ablation.py --n_jobs 8 --threads 4            (multi-core machine; 320 fits)
          python code/09_ablation.py --quick --n_jobs 2 --threads 2   (smoke test)
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



def selected_params():
    """{(map, model): params} from 08b (selection on municipality folds, per map)."""
    sel = pd.read_csv(C.TABLES_DIR / "08b_selected.csv")
    by_name = {(m, config_name(p)): p for m, grid in GRIDS.items() for p in grid}
    return {(r.pga_source, r.model): by_name[(r.model, r.config)] for r in sel.itertuples()}


def build_tasks(a):
    sel, tasks = selected_params(), []
    order = {"random_forest": 0, "lightgbm": 1, "ordinal_logit": 2, "decision_tree": 3}   # slow first

    def add(study, model, params, fs, src, level, runs):
        cfg = config_name(params)
        for scheme, seed in runs:
            for fold in range(1 if a.quick else C.N_FOLDS):
                tasks.append({"study": study, "model": model, "config": cfg, "params": params, "feature_set": fs,
                              "pga_source": src, "pga_level": level, "scheme": scheme, "seed": seed, "fold": fold,
                              "save_oof": scheme == "municipality" and seed == a.seeds[0],
                              "oof_name": f"09_{model}_{cfg}_{src}_{level}_{fs}_s{seed}"})

    muni = [("municipality", s) for s in a.seeds]
    for model in sorted(GRIDS, key=order.get):
        for src in C.SHAKEMAP_SOURCES:
            add("ablation", model, sel[src, model], "type_pga", src, "municipality", muni)
        add("ablation", model, sel[C.SHAKEMAP_PRIMARY, model], "no_pga", C.SHAKEMAP_PRIMARY, "municipality", muni)
    for label, extra, fs, level in LGBM_VARIANTS:
        for src in C.SHAKEMAP_SOURCES:
            runs = muni + ([("random", s) for s in a.random_seeds] if src == C.SHAKEMAP_PRIMARY else [])
            add("lgbm_diagnosis", "lightgbm", sel[src, "lightgbm"] | extra, fs, src, level, runs)
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=8, help="tasks run in parallel")
    p.add_argument("--threads", type=int, default=4, help="threads per task (RF, LightGBM, BLAS)")
    p.add_argument("--seeds", nargs="+", type=int, default=C.CV_REPEAT_SEEDS)
    p.add_argument("--random_seeds", nargs="+", type=int, default=[C.SEED])
    p.add_argument("--tag", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 fold, 30k/10k rows, small forests")
    a = p.parse_args()
    if a.quick:
        a.tag = f"{a.tag}_quick"

    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.TABLES_DIR / f"09_cv_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)

    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "lgbm_variants": LGBM_VARIANTS,
            "selected": {f"{k[0]}/{k[1]}": v for k, v in selected_params().items()},
            "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    seconds = run_parallel(tasks, out, a.n_jobs, a.threads, a.quick, extra_keys=("study",))

    assemble_oof({t["oof_name"] for t in build_tasks(a) if t["save_oof"]}, 1 if a.quick else C.N_FOLDS)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (C.TABLES_DIR / f"09_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))

    res = pd.read_csv(out)
    summ = (res.groupby(["study", "scheme", "pga_source", "pga_level", "model", "feature_set", "config"])
            [["rps", "brier_severe", "qwk", "best_iteration"]].agg(["mean", "std"]).round(4))
    print("\nMean and SD over folds:\n", summ.to_string())


if __name__ == "__main__":
    main()
