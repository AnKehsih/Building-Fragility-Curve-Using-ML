"""
09e_feature_importance.py

Purpose : Which inputs do the predictions depend on, and could fewer inputs do as well? For the two models
          that beat the fragility curves on new districts (ordinal logit and LightGBM boosted from the
          fragility curve, configurations of 09b_selected_final.csv), both maps, two measures per input
          group (utils.models.FEATURE_GROUPS: PGA, structural type, the 11 superstructure flags as one group,
          each other input on its own; 14 groups):
            A. Permutation importance (leave-one-district-out folds, as 09c): fit once per fold, then shuffle
               one group across the held-out district's buildings (3 repeats) and measure how much RPS
               worsens. Cheap (no refit), but understates inputs that others can stand in for (floors and
               height), and for PGA it measures only the within-district PGA spread.
            B. Drop-one-group refits: retrain without the group; the model may lean on related inputs instead,
               so this measures what the group adds that nothing else carries. Run on LODO folds (paired
               with 09c) and on municipality folds, seed 678 (paired with 08 / 09).
          Hybrid: its trees never see PGA (PGA enters through the fragility curve), so "drop pga" is not run
          for it; dropping structural type removes it from the trees only (the curve still uses it).
          Pre-declared rule (2026-11-03, before any results): a group is NEGLIGIBLE for a model and map if
          dropping it raises the mean municipality-fold RPS by less than 0.0002 (about 1/10 of the smallest
          gain of a selected model over the fragility curves on new districts, 0.0018, Step 21). The report
          lists the candidate reduced input set; fitting it is a separate, later step (selection on
          municipality folds, check on LODO, scored on the final test under D12). Not used to change the
          selection of 09b.
Inputs  : data/processed/* (as 08), results/tables/09b_selected_final.csv; for --report also
          09c_lodo_metrics_main.csv, 08_cv_metrics_main.csv, 09_cv_metrics_main.csv
Outputs : results/tables/09e_permutation_<tag>.csv, 09e_drop_<tag>.csv, 09e_run_<tag>.json
          With --report: 09e_permutation_summary.csv, 09e_drop_summary.csv, 09e_importance.csv,
          09e_console_output.txt via redirect
Run     : python code/09e_feature_importance.py --n_jobs 10 --threads 3           (multi-core machine; 850 fits)
          python code/09e_feature_importance.py --quick --n_jobs 2 --threads 2   (smoke test)
          python code/09e_feature_importance.py --report > results/tables/09e_console_output.txt
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
from utils.cvrun import KEY, run_parallel, run_permutation_task
from utils.models import DROP, FEATURE_GROUPS

T = C.TABLES_DIR
N_DISTRICTS = 10
MODELS = ("ordinal_logit", "lightgbm")
NEGLIGIBLE = 0.0002              # pre-declared, see docstring


def selected():
    """{(map, model): (params, config, feature_set)} for the two models, from 09b."""
    s = pd.read_csv(T / "09b_selected_final.csv")
    return {(r.pga_source, r.model): (json.loads(r.params), r.config, r.feature_set)
            for r in s.itertuples() if r.model in MODELS}


def build_tasks(a):
    sel, perm, drop = selected(), [], []
    n_lodo, n_muni = (1, 1) if a.quick else (N_DISTRICTS, C.N_FOLDS)
    for src in C.SHAKEMAP_SOURCES:
        for model in MODELS:
            params, cfg, fs = sel[src, model]
            base = {"model": model, "config": cfg, "params": params, "pga_source": src, "pga_level": "municipality",
                    "seed": C.SEED, "train_score": False, "save_oof": False, "oof_name": ""}
            for fold in range(n_lodo):
                perm.append(base | {"feature_set": fs, "scheme": "district", "fold": fold})
            for g in FEATURE_GROUPS:
                if model == "lightgbm" and g == "pga":
                    continue
                for scheme, n in (("district", n_lodo), ("municipality", n_muni)):
                    for fold in range(n):
                        drop.append(base | {"feature_set": f"{fs}{DROP}{g}", "group": g, "scheme": scheme, "fold": fold})
    return perm, drop


def run(a):
    from joblib import Parallel, delayed
    perm, drop = build_tasks(a)
    out_p, out_d = T / f"09e_permutation_{a.tag}.csv", T / f"09e_drop_{a.tag}.csv"
    for out in (out_p, out_d):
        if out.exists() and not a.resume:
            out.unlink()
    if a.resume and out_p.exists():
        done = set(map(tuple, pd.read_csv(out_p)[KEY].drop_duplicates().astype(str).to_numpy()))
        perm = [t for t in perm if tuple(str(t[k]) for k in KEY) not in done]
    if a.resume and out_d.exists():
        done = set(map(tuple, pd.read_csv(out_d)[KEY].astype(str).to_numpy()))
        drop = [t for t in drop if tuple(str(t[k]) for k in KEY) not in done]
    import joblib, lightgbm, sklearn
    meta = {"args": vars(a), "groups": FEATURE_GROUPS, "negligible_rule": NEGLIGIBLE,
            "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}
    print(f"A. permutation: {len(perm)} fits; B. drop-one-group: {len(drop)} fits; "
          f"{a.n_jobs} parallel x {a.threads} threads", flush=True)
    t0 = time.time()
    for i, rows in enumerate(Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
            delayed(run_permutation_task)(t, a.threads, a.quick) for t in perm), 1):
        pd.DataFrame(rows).to_csv(out_p, mode="a", header=not out_p.exists(), index=False)
        print(f"[A {i}/{len(perm)}] {rows[0]['model']} {rows[0]['pga_source']} f{rows[0]['fold']} "
              f"RPS {rows[0]['rps']:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    seconds = time.time() - t0 + run_parallel(drop, out_d, a.n_jobs, a.threads, a.quick, extra_keys=("group",))
    meta["finished"], meta["minutes"] = time.strftime("%Y-%m-%d %H:%M:%S"), round(seconds / 60, 1)
    (T / f"09e_run_{a.tag}.json").write_text(json.dumps(meta, indent=2, default=str))


def report(tag):
    pd.set_option("display.width", 220)
    sfx = "" if tag == "main" else f"_{tag}"          # quick runs must not overwrite the main report
    sel = selected()

    # A. permutation importance: shuffled minus unshuffled RPS, per fold (mean of repeats)
    p = pd.read_csv(T / f"09e_permutation_{tag}.csv")
    k = ["model", "pga_source", "fold"]
    base = p[p.group == "none"][k + ["rps"]].rename(columns={"rps": "rps_base"})
    d = p[p.group != "none"].groupby(k + ["group"])["rps"].mean().reset_index().merge(base, on=k)
    d["inc"] = d["rps"] - d["rps_base"]
    pa = d.groupby(["pga_source", "model", "group"])["inc"].agg(perm_mean="mean", perm_sd="std",
                                                                perm_folds_pos=lambda x: int((x > 0).sum())).reset_index()
    pa.to_csv(T / f"09e_permutation_summary{sfx}.csv", index=False)

    # B. drop-one-group refits, paired with the full-input model on the same folds and seed
    r = pd.read_csv(T / f"09e_drop_{tag}.csv")
    c9 = pd.read_csv(T / "09c_lodo_metrics_main.csv")
    r8, r9 = pd.read_csv(T / "08_cv_metrics_main.csv"), pd.read_csv(T / "09_cv_metrics_main.csv")
    rows = []
    for (src, model, g, scheme), a in r.groupby(["pga_source", "model", "group", "scheme"]):
        _, cfg, fs = sel[src, model]
        if scheme == "district":
            b = c9
        else:
            b = r8 if model == "ordinal_logit" else r9
            b = b[b.seed == C.SEED]
        b = b[(b.model == model) & (b.config == cfg) & (b.feature_set == fs) & (b.pga_source == src)
              & (b.pga_level == "municipality") & (b.scheme == scheme)]
        m = a[["fold", "rps"]].merge(b[["fold", "rps"]], on="fold", suffixes=("", "_full"))
        x = m.rps - m.rps_full
        rows.append({"pga_source": src, "model": model, "group": g, "scheme": scheme, "drop_mean": x.mean(),
                     "drop_sd": x.std(), "drop_folds_worse": int((x > 0).sum()), "n_folds": len(x)})
    dr = pd.DataFrame(rows)
    dr.to_csv(T / f"09e_drop_summary{sfx}.csv", index=False)

    # combined table: one row per map, model, group
    w = dr.pivot_table(index=["pga_source", "model", "group"], columns="scheme",
                       values=["drop_mean", "drop_folds_worse"]).reset_index()
    w.columns = ["_".join(x for x in col if x).replace("district", "lodo").replace("municipality", "muni")
                 for col in w.columns]
    imp = pa.merge(w, on=["pga_source", "model", "group"], how="outer")
    imp["negligible"] = imp["drop_mean_muni"] < NEGLIGIBLE
    imp = imp.sort_values(["pga_source", "model", "perm_mean"], ascending=[True, True, False])
    imp.to_csv(T / f"09e_importance{sfx}.csv", index=False)
    print("Input importance (RPS increase; positive = the input helps). perm_*: shuffle the group in the held-out")
    print("district (LODO, 10 folds, 3 repeats). drop_*: refit without the group (lodo: 10 districts, paired with")
    print(f"09c; muni: municipality folds seed {C.SEED}, paired with 08/09). negligible: drop_mean_muni < {NEGLIGIBLE}")
    print("(rule fixed before the run). Hybrid: no 'pga' drop (PGA is not a tree input).")
    for (src, model), g in imp.groupby(["pga_source", "model"]):
        print(f"\n{model}, {src}:")
        print(g.drop(columns=["pga_source", "model"]).round(5).to_string(index=False))
    print("\nCandidate reduced input sets (negligible groups to drop; to be fitted and checked in a later step):")
    for (src, model), g in imp.groupby(["pga_source", "model"]):
        print(f"  {model}, {src}: drop {sorted(g.loc[g.negligible, 'group'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=10)
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
