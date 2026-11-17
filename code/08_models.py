"""
08_models.py

Purpose : Cross-validated ML models on the 10 development districts (CONTEXT 4, 5.2):
            ordinal_logit (ridge, own fitter in utils/ordinal.py), decision_tree (shallow),
            random_forest, lightgbm (early stopping on an inner split of the training fold).
          Each model x hyperparameter configuration (utils/models.GRIDS) x PGA source is fitted on
          the training folds and scored on the held-out folds fixed by 06_splits.py, so results
          are paired with the baselines of 07. Feature set "full" by default (all allowed
          pre-earthquake inputs + structural type + ln PGA); "type_pga" is used by 09 (ablation).
          Municipality folds: all PGA sources, all seeds. Random folds (comparison only): primary
          PGA source, --random_seeds.
          Runs tasks in parallel (--n_jobs processes x --threads threads each). Every finished task
          is appended to the metrics CSV at once; --resume skips tasks already there.
Inputs  : data/processed/buildings_clean.parquet, pga_municipality.csv, pga_district.csv,
          folds_municipality.csv, folds_random.parquet
Outputs : results/tables/08_cv_metrics_<tag>.csv    one row per model x config x PGA x scheme x seed x fold
          results/tables/08_run_<tag>.json          arguments, grids, package versions, run time
          results/oof/08_<model>_<config>_<pga>_<fs>_s<seed>.parquet   out-of-fold P (municipality folds,
                                                    first seed): building_id, fold, p1..p5
Run     : python code/08_models.py --n_jobs 8 --threads 4            (8 x 4 = 32 cores)
          python code/08_models.py --quick --n_jobs 2 --threads 2   (smoke test on the laptop, ~2 min)
          Options: --models, --pga_sources, --feature_set, --seeds, --random_seeds, --tag, --resume
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.cvrun import KEY, assemble_oof, run_task
from utils.models import GRIDS, FEATURE_SETS, config_name


def build_tasks(a):
    tasks = []
    order = {"random_forest": 0, "lightgbm": 1, "ordinal_logit": 2, "decision_tree": 3}   # slow first
    for model in sorted(a.models, key=order.get):
        for params in GRIDS[model]:
            cfg = config_name(params)
            runs = [("municipality", src, seed) for src in a.pga_sources for seed in a.seeds]
            runs += [("random", C.SHAKEMAP_PRIMARY, seed) for seed in a.random_seeds]
            for scheme, src, seed in runs:
                for fold in range(1 if a.quick else C.N_FOLDS):
                    save = scheme == "municipality" and seed == a.seeds[0]
                    tasks.append({"model": model, "config": cfg, "params": params, "feature_set": a.feature_set,
                                  "pga_source": src, "pga_level": "municipality", "scheme": scheme, "seed": seed, "fold": fold,
                                  "save_oof": save, "oof_name": f"08_{model}_{cfg}_{src}_{a.feature_set}_s{seed}"})
    return tasks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=8, help="tasks run in parallel")
    p.add_argument("--threads", type=int, default=4, help="threads per task (RF, LightGBM, BLAS)")
    p.add_argument("--models", nargs="+", default=list(GRIDS))
    p.add_argument("--pga_sources", nargs="+", default=[C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY])
    p.add_argument("--feature_set", default="full", choices=list(FEATURE_SETS))
    p.add_argument("--seeds", nargs="+", type=int, default=C.CV_REPEAT_SEEDS)
    p.add_argument("--random_seeds", nargs="+", type=int, default=[C.SEED])
    p.add_argument("--tag", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 fold, 30k/10k rows, small forests")
    a = p.parse_args()
    if a.quick:
        a.tag = f"{a.tag}_quick"

    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = C.TABLES_DIR / f"08_cv_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)

    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "grids": GRIDS, "python": platform.python_version(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0, n_done = time.time(), 0
    jobs = Parallel(n_jobs=a.n_jobs, return_as="generator_unordered", verbose=0)(
        delayed(run_task)(t, a.threads, a.quick) for t in tasks)
    for row in jobs:
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists() or out.stat().st_size == 0, index=False)
        n_done += 1
        el = time.time() - t0
        print(f"[{n_done}/{len(tasks)}] {row['model']:13s} {row['config']:28s} {row['pga_source']:9s} "
              f"{row['scheme']:12s} s{row['seed']} f{row['fold']}  RPS {row['rps']:.4f}  "
              f"{row['fit_seconds']:6.1f}s  ETA {el / n_done * (len(tasks) - n_done) / 60:5.1f} min", flush=True)

    assemble_oof({t['oof_name'] for t in build_tasks(a) if t['save_oof']}, 1 if a.quick else C.N_FOLDS)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round((time.time() - t0) / 60, 1)
    (C.TABLES_DIR / f"08_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))

    res = pd.read_csv(out)
    summ = (res.groupby(["scheme", "pga_source", "model", "config"])[["rps", "brier_severe", "qwk", "macro_f1", "accuracy"]]
            .agg(["mean", "std"]).round(4))
    print("\nMean and SD over folds:\n", summ.to_string())


if __name__ == "__main__":
    main()
