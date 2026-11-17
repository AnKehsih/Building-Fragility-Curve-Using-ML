"""
Worker side of the parallel CV runs (08, 09, 10). Lives in a module (not in a numbered script)
so that joblib worker processes can import it. Each worker loads the data once (cached).

A task is a dict with: model, config, params, feature_set, pga_source, pga_level, scheme, seed,
fold, save_oof, oof_name. run_task fits on the training fold only and scores the held-out fold.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils.cv import fold_ids
from utils.data import load_development, attach_pga
from utils.metrics import evaluate
from utils.models import fit_predict
from utils.preprocessing import assign_structural_type

OOF_DIR = C.PROJECT_ROOT / "results" / "oof"
KEY = ["model", "config", "feature_set", "pga_source", "pga_level", "scheme", "seed", "fold"]
TRAIN_SCORE_ROWS = 100_000      # training rows scored when a task asks for train_score


@lru_cache(maxsize=4)
def model_data(pga_source, pga_level="municipality"):
    """Development rows with structural_type and ln_pga. Cached per worker process."""
    df = attach_pga(load_development(), pga_source, pga_level)
    df["structural_type"] = assign_structural_type(df, C.STRUCTURAL_TYPE_RULE).astype(str)
    df["ln_pga"] = np.log(df["pga"])
    return df


@lru_cache(maxsize=8)
def folds(scheme, seed):
    return fold_ids(load_development(), scheme, seed)


def run_task(t, threads, quick=False):
    df = model_data(t["pga_source"], t["pga_level"])
    f = folds(t["scheme"], t["seed"])
    tr, va = np.flatnonzero(f != t["fold"]), np.flatnonzero(f == t["fold"])
    if quick:                                     # smoke test: small random subsets
        rng = np.random.default_rng(0)
        tr, va = np.sort(rng.choice(tr, 30000, replace=False)), np.sort(rng.choice(va, 10000, replace=False))
    y = df[C.TARGET].to_numpy()
    # grouped schemes: LightGBM's early-stopping set = whole municipalities of the training fold
    groups = df[C.GROUP_COL].to_numpy()[tr] if t["scheme"] != "random" else None
    # optional: also score a random subsample of the TRAINING rows (train-validation gap = overfitting)
    sub = None
    if t.get("train_score"):
        sub = np.sort(np.random.default_rng(t["fold"]).choice(len(tr), min(TRAIN_SCORE_ROWS, len(tr)), replace=False))
    P, info = fit_predict(t["model"], t["params"], t["feature_set"], df.iloc[tr], df.iloc[va], y[tr],
                          threads, seed=t["seed"] * 10 + t["fold"], groups=groups, quick=quick, train_rows=sub)
    P_train, P_inbag = info.pop("P_train", None), info.pop("P_train_inbag", None)
    row = {k: t[k] for k in KEY} | {"n_train": len(tr)} | evaluate(P, y[va]) | info
    if P_train is not None:      # random forest: out-of-bag (see fit_predict); in-bag RPS kept for comparison
        row |= {f"train_{k}": v for k, v in evaluate(P_train, y[tr][sub]).items() if k != "n"}
    if P_inbag is not None:
        row["train_inbag_rps"] = evaluate(P_inbag, y[tr][sub])["rps"]
    if t["save_oof"]:
        part = pd.DataFrame(P.astype("float32"), columns=[f"p{g}" for g in C.GRADES])
        part.insert(0, "fold", t["fold"])
        part.insert(0, "building_id", df["building_id"].to_numpy()[va])
        d = OOF_DIR / "parts" / t["oof_name"]
        d.mkdir(parents=True, exist_ok=True)
        part.to_parquet(d / f"fold{t['fold']}.parquet", index=False)
    return row


PERM_REPEATS = 3                # permutations per input group (09e)


def run_permutation_task(t, threads, quick=False):
    """Permutation importance (09e): fit on the training fold as run_task does, score the held-out fold,
    then re-score it with each input group of FEATURE_GROUPS shuffled across the held-out buildings (all
    columns of a group with the same shuffle, so e.g. flag combinations stay intact). No refitting.
    Returns one row per (group, repeat); group "none" is the unshuffled score."""
    from utils.models import FEATURE_GROUPS, _norm, make_model
    df = model_data(t["pga_source"], t["pga_level"])
    f = folds(t["scheme"], t["seed"])
    tr, va = np.flatnonzero(f != t["fold"]), np.flatnonzero(f == t["fold"])
    if quick:
        rng = np.random.default_rng(0)
        tr, va = np.sort(rng.choice(tr, 30000, replace=False)), np.sort(rng.choice(va, 10000, replace=False))
    y = df[C.TARGET].to_numpy()
    groups = df[C.GROUP_COL].to_numpy()[tr] if t["scheme"] != "random" else None
    m = make_model(t["model"], t["params"], t["feature_set"], threads, t["seed"] * 10 + t["fold"], quick)
    m.fit(df.iloc[tr], y[tr], groups)
    val = df.iloc[va].reset_index(drop=True)
    head = {k: t[k] for k in KEY}
    keep = ("rps", "brier_severe", "log_loss")
    base = evaluate(_norm(m.predict_proba(val)), y[va])
    rows = [head | {"group": "none", "repeat": -1} | {k: base[k] for k in keep}]
    rng = np.random.default_rng(t["seed"] * 100 + t["fold"])
    for g, cols in FEATURE_GROUPS.items():
        for rep in range(PERM_REPEATS):
            v = val.copy()
            v[cols] = val[cols].iloc[rng.permutation(len(val))].reset_index(drop=True)
            s = evaluate(_norm(m.predict_proba(v)), y[va])
            rows.append(head | {"group": g, "repeat": rep} | {k: s[k] for k in keep})
    return rows


def run_parallel(tasks, out, n_jobs, threads, quick=False, extra_keys=()):
    """Run tasks with joblib; append each finished row to the CSV `out` at once (safe to interrupt).
    `extra_keys`: task fields copied into each row (e.g. "study")."""
    import time
    from joblib import Parallel, delayed
    t0, n_done = time.time(), 0
    rows = Parallel(n_jobs=n_jobs, return_as="generator_unordered", verbose=0)(
        delayed(_run_with_extra)(t, threads, quick, extra_keys) for t in tasks)
    for row in rows:
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists() or out.stat().st_size == 0, index=False)
        n_done += 1
        el = time.time() - t0
        print(f"[{n_done}/{len(tasks)}] {row['model']:13s} {row['config']:28s} {row['feature_set']:8s} "
              f"{row['pga_source']:9s} {row['pga_level'][:4]} {row['scheme']:12s} s{row['seed']} f{row['fold']}  "
              f"RPS {row['rps']:.4f}  {row['fit_seconds']:6.1f}s  ETA {el / n_done * (len(tasks) - n_done) / 60:5.1f} min",
              flush=True)
    return time.time() - t0


def _run_with_extra(t, threads, quick, extra_keys):
    return {k: t[k] for k in extra_keys} | run_task(t, threads, quick)


def assemble_oof(names, n_folds):
    """Join per-fold out-of-fold parts into one file per model run (only when all folds exist)."""
    for name in sorted(names):
        parts = sorted((OOF_DIR / "parts" / name).glob("fold*.parquet"))
        if len(parts) == n_folds:
            pd.concat(pd.read_parquet(p) for p in parts).to_parquet(OOF_DIR / f"{name}.parquet", index=False)
