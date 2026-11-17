"""
09f_reduced_inputs.py

Purpose : Post-hoc (after 09e, Step 25; NOT used to change the 09b selection): would a survey that collects
          fewer attributes predict as well? 09e dropped one input group at a time; six groups were
          negligible for both selected models on both maps (utils.models.REDUCED_DROP: ground floor type,
          height, land surface condition, plan configuration, plinth area, position). Groups can stand in
          for each other, so here all six are dropped TOGETHER and the two selected models (ordinal logit;
          LightGBM boosted from the fragility curve; configurations of 09b_selected_final.csv) are refitted:
            municipality folds, 5 folds x 3 seeds   paired with 08 (ordinal logit) / 09 (hybrid), full inputs
            leave-one-district-out, 10 folds        paired with 09c (full inputs, fragility curves), with the
                                                    train-validation gap as in 09c
          Structural type stays in both models: the fragility curve (and so the hybrid) needs it, so a
          survey must collect it anyway.
          Pre-declared rule (2026-11-03, before any results): the reduced set is ACCEPTED for a model and map
          if dropping all six groups together raises the mean municipality-fold RPS (15 folds) by less than
          0.0005 (about 1/4 of the smallest gain of a selected model over the fragility curves on new
          districts, 0.0018, Step 21). The LODO rows are a check, reported next to it (incl. the
          ground-floor-type concern of Step 25), not part of the rule. Under D12 the reduced models are also
          scored on the final test, labelled post-hoc.
Inputs  : data/processed/* (as 08), results/tables/09b_selected_final.csv; for --report also
          08_cv_metrics_main.csv, 09_cv_metrics_main.csv, 09c_lodo_metrics_main.csv, 07_cv_metrics.csv
Outputs : results/tables/09f_metrics_<tag>.csv, 09f_run_<tag>.json
          With --report: 09f_summary.csv, 09f_console_output.txt via redirect
Run     : python code/09f_reduced_inputs.py --n_jobs 8 --threads 3            (multi-core machine; 100 fits)
          python code/09f_reduced_inputs.py --quick --n_jobs 2 --threads 2   (smoke test)
          python code/09f_reduced_inputs.py --report > results/tables/09f_console_output.txt
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
from utils.models import REDUCED_DROP

T = C.TABLES_DIR
N_DISTRICTS = 10
MODELS = ("ordinal_logit", "lightgbm")
ACCEPT = 0.0005                  # pre-declared, see docstring


def selected():
    """{(map, model): (params, config, feature_set)} for the two models, from 09b."""
    s = pd.read_csv(T / "09b_selected_final.csv")
    return {(r.pga_source, r.model): (json.loads(r.params), r.config, r.feature_set)
            for r in s.itertuples() if r.model in MODELS}


def build_tasks(a):
    sel, tasks = selected(), []
    seeds = C.CV_REPEAT_SEEDS[:1] if a.quick else C.CV_REPEAT_SEEDS
    n_muni, n_lodo = (1, 1) if a.quick else (C.N_FOLDS, N_DISTRICTS)
    for src in C.SHAKEMAP_SOURCES:
        for model in MODELS:
            params, cfg, fs = sel[src, model]
            # every task scores training rows, so all rows of the output CSV have the same columns
            base = {"model": model, "config": cfg, "params": params, "feature_set": f"{fs}_reduced",
                    "pga_source": src, "pga_level": "municipality", "train_score": True, "save_oof": False,
                    "oof_name": ""}
            tasks += [base | {"scheme": "municipality", "seed": s, "fold": f} for s in seeds for f in range(n_muni)]
            tasks += [base | {"scheme": "district", "seed": C.SEED, "fold": f} for f in range(n_lodo)]
    return tasks


def run(a):
    out = T / f"09f_metrics_{a.tag}.csv"
    tasks = build_tasks(a)
    if a.resume and out.exists():
        done = set(map(tuple, pd.read_csv(out)[KEY].astype(str).to_numpy()))
        tasks = [t for t in tasks if tuple(str(t[k]) for k in KEY) not in done]
    elif out.exists():
        out.unlink()
    print(f"{len(tasks)} tasks; {a.n_jobs} parallel x {a.threads} threads; writing {out}", flush=True)
    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "reduced_drop": REDUCED_DROP, "accept_rule": ACCEPT, "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    seconds = run_parallel(tasks, out, a.n_jobs, a.threads, a.quick)
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (T / f"09f_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))


def report(tag):
    """Pair the reduced models with the full-input models and the fragility curves, fold by fold."""
    pd.set_option("display.width", 220)
    sfx = "" if tag == "main" else f"_{tag}"          # quick runs must not overwrite the main report
    sel = selected()
    r = pd.read_csv(T / f"09f_metrics_{tag}.csv")
    full_muni = pd.concat([pd.read_csv(T / "08_cv_metrics_main.csv"), pd.read_csv(T / "09_cv_metrics_main.csv")])
    c9 = pd.read_csv(T / "09c_lodo_metrics_main.csv")
    b7 = pd.read_csv(T / "07_cv_metrics.csv")
    frag_lodo = c9[(c9.model == "fragility") & (c9.pga_level == "municipality")]
    frag_lodo = frag_lodo.assign(frag_gap=frag_lodo.rps - frag_lodo.train_rps)
    k = ["seed", "fold"]
    rows = []
    for (src, model, scheme), a in r.groupby(["pga_source", "model", "scheme"]):
        _, cfg, fs = sel[src, model]
        full = c9 if scheme == "district" else full_muni
        full = full[(full.model == model) & (full.config == cfg) & (full.feature_set == fs) & (full.pga_source == src)
                    & (full.pga_level == "municipality") & (full.scheme == scheme)].drop_duplicates(k)
        if scheme == "district":
            frag = frag_lodo[frag_lodo.pga_source == src]
        else:
            frag = b7[(b7.model == f"fragility_{src}_municipality") & (b7.scheme == scheme)].assign(frag_gap=np.nan)
        m = (a[k + ["rps", "brier_severe", "train_rps"]]
             .merge(full[k + ["rps", "brier_severe"]], on=k, suffixes=("", "_full"))
             .merge(frag[k + ["rps", "frag_gap"]].rename(columns={"rps": "rps_frag"}), on=k))
        d_full, d_frag = m.rps - m.rps_full, m.rps - m.rps_frag
        gap = (m.rps - m.train_rps) - m.frag_gap
        rows.append({"pga_source": src, "model": model, "scheme": scheme, "n_folds": len(m),
                     "rps_reduced": m.rps.mean(), "rps_full": m.rps_full.mean(),
                     "d_vs_full": d_full.mean(), "d_vs_full_se": d_full.std() / np.sqrt(len(m)),
                     "folds_worse_than_full": int((d_full > 0).sum()),
                     "d_brier_severe_vs_full": (m.brier_severe - m.brier_severe_full).mean(),
                     "d_vs_fragility": d_frag.mean(), "folds_better_than_fragility": int((d_frag < 0).sum()),
                     "excess_gap": gap.mean()})
    s = pd.DataFrame(rows)
    acc = s[s.scheme == "municipality"].set_index(["pga_source", "model"]).d_vs_full < ACCEPT
    s["accepted"] = [acc.get((x.pga_source, x.model)) for x in s.itertuples()]
    s.to_csv(T / f"09f_summary{sfx}.csv", index=False)
    print(f"Reduced input set: all inputs except {REDUCED_DROP}, dropped together.")
    print("d_vs_full: reduced minus full-input RPS, paired fold by fold (positive = the reduced set is worse);")
    print("municipality: 5 folds x 3 seeds, paired with 08/09; district: 10 held-out districts, paired with 09c.")
    print("d_vs_fragility: reduced minus fragility curves (same map, municipality PGA; negative = better).")
    print(f"excess_gap (district only): train-validation gap minus the fragility curves' gap. accepted:")
    print(f"municipality-fold d_vs_full < {ACCEPT} (rule fixed before the run).\n")
    print(s.round(5).to_string(index=False))


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
