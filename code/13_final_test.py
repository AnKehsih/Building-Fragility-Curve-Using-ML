"""
13_final_test.py

Purpose : The single, final evaluation on the held-out district (Kavrepalanchok, D5). Under D12 every trained
          model is refitted on ALL development buildings and scored ONCE on the test district; the models
          selected on validation (D9/D11/D13, 09b_selected_final.csv) are marked before the test and are the
          headline. Test scores never change the selection.
          Four modes, in this order:
            --list      build the run list (no test data) -> 13_freeze_list_DRAFT.csv, for the student to review
            --freeze    STUDENT ONLY, after review: write 13_freeze_list.csv and 13_freeze.json (time, the
                        pre-declared headline rule, SHA-256 of the list, of the code and of the selection tables)
            --dry_run   whole pipeline with a DEVELOPMENT district (Sindhuli) standing in for the test district
                        (fitted on the other 9); Kavrepalanchok is not read. Use with --quick for a smoke test.
            --run       the final test. Refuses unless the freeze exists and every hash still matches (nothing
                        changed since the freeze), and refuses to run twice (--resume only completes missing
                        runs of an interrupted run). Then --report.
          Run list (unique by model, configuration, feature set, map, PGA level):
            baselines     fragility curves (municipality and district PGA, both maps), majority
            08 grid       every configuration of the four ML models, full inputs, municipality PGA, both maps
            09 ablation   type_pga (both maps) and no_pga (preferred map) at the 08b configurations
            09 LightGBM   district_pga, pga_bins5, pga_bins10, init_fragility (hybrid), both maps
            post-hoc      09d (forest min leaf 500 / 2000, LightGBM 10/20/40/100 rounds), 09f (reduced inputs),
                          10 (district PGA for the ordinal logit and the hybrid); labelled post-hoc
          Pre-declared headline rule (fixed at --freeze, see HEADLINE_RULE).
Inputs  : data/processed/*, results/tables/08b_selected.csv, 09b_selected_final.csv, and the validation tables
          (07, 08, 09, 09c, 09d, 09f, 10) for the validation-vs-test comparison in --report
Outputs : results/tables/13_freeze_list_DRAFT.csv; 13_freeze_list.csv, 13_freeze.json (--freeze);
          13_test_metrics.csv, 13_run.json, results/oof/13_<run_id>.parquet (--run);
          13_headline.csv, 13_all_runs.csv, 13_console_output.txt via redirect (--report)
Run     : python code/13_final_test.py --list
          python code/13_final_test.py --freeze                             (student, after review)
          python code/13_final_test.py --dry_run --quick --n_jobs 4 --threads 2
          python code/13_final_test.py --run --n_jobs 8 --threads 3          (multi-core machine; once)
          python code/13_final_test.py --report > results/tables/13_console_output.txt
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import attach_pga, load_development, load_final_test
from utils.metrics import evaluate, p_severe, rps
from utils.models import GRIDS, LGBM_VARIANTS, config_name, fit_predict
from utils.preprocessing import assign_structural_type

T = C.TABLES_DIR
OOF_DIR = C.PROJECT_ROOT / "results" / "oof"
SPEC = ["model", "config", "feature_set", "pga_source", "pga_level"]
DRY_RUN_DISTRICT = 20             # Sindhuli: development district standing in for the test district in --dry_run
N_BOOT = 2000
HEADLINE_RULE = (
    "Headline (fixed before the test): for each map, each of the four validation-selected models "
    "(09b_selected_final.csv) against the fragility curves of the same map (municipality PGA), paired building "
    "by building on the test district. Primary metric RPS; also severe-damage Brier score and log loss. An "
    "improvement is claimed only if the mean paired RPS difference is negative AND its 95% cluster-bootstrap "
    "interval over the test district's 13 municipalities (2,000 resamples) excludes zero; otherwise 'no clear "
    "difference'. The number of municipalities where the model is better is reported alongside. All other runs "
    "are reported in a secondary table; the best test score among them is not a result. Post-hoc runs are "
    "labelled post-hoc.")
FROZEN_FILES = ["code/13_final_test.py", "code/config.py", "code/utils/models.py", "code/utils/ordinal.py",
                "code/utils/fragility.py", "code/utils/data.py", "code/utils/preprocessing.py",
                "code/utils/metrics.py", "results/tables/08b_selected.csv", "results/tables/09b_selected_final.csv"]


# ---------------------------------------------------------------- run list
def run_list():
    by_name = {(m, config_name(p)): p for m, g in GRIDS.items() for p in g}
    s8 = pd.read_csv(T / "08b_selected.csv")
    s8 = {(r.pga_source, r.model): by_name[r.model, r.config] for r in s8[s8.scheme == "municipality"].itertuples()}
    s9 = pd.read_csv(T / "09b_selected_final.csv")
    sel = {(r.model, r.config, r.feature_set, r.pga_source, r.pga_level) for r in s9.itertuples()}
    rows = []

    def add(group, model, params, fs, src, level):
        rows.append({"group": group, "model": model, "config": config_name(params) if params else "-",
                     "params": json.dumps(params), "feature_set": fs, "pga_source": src, "pga_level": level})

    for src in C.SHAKEMAP_SOURCES:
        for level in ("municipality", "district"):
            add("baseline", "fragility", {}, "type_pga", src, level)
        for model, grid in GRIDS.items():
            for p in grid:
                add("08 grid", model, p, "full", src, "municipality")
            add("09 ablation", model, s8[src, model], "type_pga", src, "municipality")
            if src == C.SHAKEMAP_PRIMARY:
                add("09 ablation", model, s8[src, model], "no_pga", src, "municipality")
        for _, extra, fs, level in LGBM_VARIANTS:
            add("09 LightGBM variant", "lightgbm", s8[src, "lightgbm"] | extra, fs, src, level)
        for leaf in (500, 2000):
            add("post-hoc 09d", "random_forest", {"min_samples_leaf": leaf}, "full", src, "municipality")
        for n in (10, 20, 40, 100):
            add("post-hoc 09d", "lightgbm", s8[src, "lightgbm"] | {"n_rounds": n}, "full", src, "municipality")
        for r in s9[(s9.pga_source == src) & s9.model.isin(["ordinal_logit", "lightgbm"])].itertuples():
            p = json.loads(r.params)
            add("post-hoc 09f", r.model, p, f"{r.feature_set}_reduced", src, "municipality")
            add("post-hoc 10", r.model, p, r.feature_set, src, "district")
    add("baseline", "majority", {}, "type_pga", C.SHAKEMAP_PRIMARY, "municipality")
    d = pd.DataFrame(rows)
    d = d.groupby(SPEC, sort=False).agg(groups=("group", " + ".join), params=("params", "first")).reset_index()
    d["selected"] = [tuple(r) in sel for r in d[SPEC].itertuples(index=False)]
    d.loc[d.model.isin(["fragility", "majority"]), "role"] = "baseline"
    d.loc[(d.model == "fragility") & (d.pga_level == "municipality"), "role"] = "headline baseline"
    d.loc[d.selected, "role"] = "headline (validation-selected)"
    d["role"] = d.role.fillna(pd.Series(np.where(d.groups.str.contains("post-hoc"), "post-hoc", "secondary"),
                                        index=d.index))
    d.insert(0, "run_id", [f"r{i:02d}" for i in range(len(d))])
    return d


def sha(path):
    return hashlib.sha256(Path(C.PROJECT_ROOT / path).read_bytes()).hexdigest()


def freeze():
    lst = run_list()
    out = T / "13_freeze_list.csv"
    if out.exists():
        sys.exit(f"{out.name} exists: the freeze is done once. Delete it only with a logged reason.")
    lst.to_csv(out, index=False)
    meta = {"frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"), "headline_rule": HEADLINE_RULE,
            "n_runs": len(lst), "sha256": {p: sha(p) for p in FROZEN_FILES + [str(out.relative_to(C.PROJECT_ROOT))]}}
    (T / "13_freeze.json").write_text(json.dumps(meta, indent=2))
    print(f"Frozen {len(lst)} runs ({int(lst.selected.sum())} validation-selected). Log the freeze in PROGRESS.md.")


def check_freeze():
    f = T / "13_freeze.json"
    if not f.exists():
        sys.exit("No freeze: run --freeze (student, after reviewing the draft list) and log it in PROGRESS.md.")
    meta = json.loads(f.read_text())
    changed = [p for p, h in meta["sha256"].items() if sha(p) != h]
    if changed:
        sys.exit(f"Changed since the freeze: {changed}. The final test must use the frozen code and selection.")
    return pd.read_csv(T / "13_freeze_list.csv"), meta


# ---------------------------------------------------------------- data and one run
def _prepare(df, src, level):
    df = attach_pga(df, src, level)
    df["structural_type"] = assign_structural_type(df, C.STRUCTURAL_TYPE_RULE).astype(str)
    df["ln_pga"] = np.log(df["pga"])
    return df


_DATA = {}                        # per-process cache (a plain dict: an lru_cache wrapper does not unpickle in workers)


def data(src, level, mode, quick):
    """(train, test) frames. mode 'final': all development vs the test district; 'dry': the other 9 development
    districts vs DRY_RUN_DISTRICT (the test district is not read)."""
    key = (src, level, mode, quick)
    if key not in _DATA:
        _DATA[key] = _load(src, level, mode, quick)
    return _DATA[key]


def _load(src, level, mode, quick):
    dev = load_development()
    if mode == "final":
        train, test = dev, load_final_test(models_frozen=True)
    else:
        train, test = dev[dev.district_id != DRY_RUN_DISTRICT], dev[dev.district_id == DRY_RUN_DISTRICT]
    if quick:
        train, test = train.sample(30_000, random_state=C.SEED), test.sample(10_000, random_state=C.SEED)
    return _prepare(train.reset_index(drop=True), src, level), _prepare(test.reset_index(drop=True), src, level)


def one_run(r, mode, threads, quick, tag):
    """r: one row of the run list as a dict (dicts pickle to joblib workers; itertuples rows do not)."""
    train, test = data(r["pga_source"], r["pga_level"], mode, quick)
    y_tr, y_te = train[C.TARGET].to_numpy(int), test[C.TARGET].to_numpy(int)
    P, info = fit_predict(r["model"], json.loads(r["params"]), r["feature_set"], train, test, y_tr, threads,
                          seed=C.SEED, groups=train[C.GROUP_COL].to_numpy(), quick=quick)
    part = pd.DataFrame(P.astype("float32"), columns=[f"p{g}" for g in C.GRADES])
    part.insert(0, "building_id", test["building_id"].to_numpy())
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    part.to_parquet(OOF_DIR / f"13_{tag}_{r['run_id']}.parquet", index=False)
    return {"run_id": r["run_id"]} | {k: r[k] for k in SPEC} | {"n_train": len(train)} \
        | evaluate(P, y_te) | {k: info[k] for k in ("fit_seconds", "best_iteration")}


def run(a, mode):
    from joblib import Parallel, delayed
    if mode == "final":
        lst, meta = check_freeze()
        tag = "final"
    else:
        lst, meta, tag = run_list(), {}, "dry_quick" if a.quick else "dry"
    out = T / f"13_test_metrics{'' if mode == 'final' else '_' + tag}.csv"
    if out.exists() and mode == "final" and not a.resume:
        sys.exit(f"{out.name} exists: the final test runs once. Use --resume only to complete an interrupted run.")
    if out.exists() and (mode != "final" or a.resume):
        if mode == "final":
            lst = lst[~lst.run_id.isin(pd.read_csv(out).run_id)]
        else:
            out.unlink()
    print(f"{mode}: {len(lst)} runs; {a.n_jobs} parallel x {a.threads} threads", flush=True)
    t0 = time.time()
    order = lst.assign(o=lst.model.map({"random_forest": 0, "lightgbm": 1}).fillna(2)).sort_values("o")
    for i, row in enumerate(Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
            delayed(one_run)(r, mode, a.threads, a.quick, tag) for r in order.to_dict("records")), 1):
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"[{i}/{len(lst)}] {row['run_id']} {row['model']:13s} {row['config'][:40]:40s} {row['feature_set']:18s} "
              f"{row['pga_source']:9s} {row['pga_level'][:4]}  RPS {row['rps']:.4f}  {(time.time() - t0) / 60:.1f} min",
              flush=True)
    import joblib, lightgbm, sklearn
    (T / f"13_run{'' if mode == 'final' else '_' + tag}.json").write_text(json.dumps(
        {"mode": mode, "args": vars(a), "freeze": meta.get("frozen_at"), "python": platform.python_version(),
         "host": platform.node(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
         "minutes": round((time.time() - t0) / 60, 1),
         "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                      "lightgbm": lightgbm.__version__, "joblib": joblib.__version__}}, indent=2))


# ---------------------------------------------------------------- report
def validation_scores():
    """Mean municipality-fold RPS (07/08/09/09f/10) and leave-one-district-out RPS (09c/09d/09f/10) per spec."""
    muni, lodo = [], []
    for f in ["08_cv_metrics_main", "09_cv_metrics_main", "09f_metrics_main", "10_metrics_main"]:
        d = pd.read_csv(T / f"{f}.csv")
        muni.append(d[d.scheme == "municipality"])
        lodo.append(d[d.scheme == "district"])
    for f in ["09c_lodo_metrics_main", "09d_metrics_main"]:
        lodo.append(pd.read_csv(T / f"{f}.csv"))
    b7 = pd.read_csv(T / "07_cv_metrics.csv")
    b7 = b7[b7.model.str.startswith("fragility") & (b7.scheme == "municipality")].copy()
    parts = b7.model.str.extract(r"fragility_(\w+?)_(municipality|district)$")
    b7 = b7.assign(model="fragility", config="-", feature_set="type_pga", pga_source=parts[0], pga_level=parts[1])
    muni.append(b7)
    m = pd.concat(muni).groupby(SPEC).rps.mean().rename("val_rps_municipality_folds")
    lo = pd.concat(lodo)
    lo = lo.assign(config=lo.config.fillna("-")).groupby(SPEC).rps.mean().rename("val_rps_lodo")
    return pd.concat([m, lo], axis=1).reset_index()


def report(mode, quick):
    from scipy.stats import spearmanr
    pd.set_option("display.width", 220)
    tag = "final" if mode == "final" else ("dry_quick" if quick else "dry")
    sfx = "" if mode == "final" else f"_{tag}"
    lst = check_freeze()[0] if mode == "final" else run_list()
    res = pd.read_csv(T / f"13_test_metrics{sfx}.csv").merge(lst[["run_id", "groups", "role", "selected"]], on="run_id")
    allr = res.merge(validation_scores(), on=SPEC, how="left")
    allr.to_csv(T / f"13_all_runs{sfx}.csv", index=False)

    # headline: paired with the fragility curves of the same map, building by building
    train, test = data(C.SHAKEMAP_PRIMARY, "municipality", "final" if mode == "final" else "dry", quick)
    t = test[["building_id", C.TARGET, C.GROUP_COL]].set_index("building_id")
    y = t[C.TARGET].to_numpy(int)

    def per_building(run_id):
        P = pd.read_parquet(OOF_DIR / f"13_{tag}_{run_id}.parquet").set_index("building_id").loc[t.index]
        P = P[[f"p{g}" for g in C.GRADES]].to_numpy(float)
        return rps(P, y), (p_severe(P) - (y >= C.SEVERE_MIN_GRADE)) ** 2, p_severe(P)

    rng = np.random.default_rng(C.SEED)
    muni_codes, groups = np.unique(t[C.GROUP_COL].to_numpy(), return_inverse=True)
    idx = rng.integers(0, len(muni_codes), (N_BOOT, len(muni_codes)))
    rows = []
    for src in C.SHAKEMAP_SOURCES:
        f_id = res[(res.model == "fragility") & (res.pga_source == src) & (res.pga_level == "municipality")].run_id.iloc[0]
        f_rps, f_bs, f_ps = per_building(f_id)
        for r in res[res.selected & (res.pga_source == src)].itertuples():
            m_rps, m_bs, m_ps = per_building(r.run_id)
            row = {"pga_source": src, "model": r.model, "config": r.config, "feature_set": r.feature_set,
                   "rps": m_rps.mean(), "rps_fragility": f_rps.mean()}
            for name, d in (("rps", m_rps - f_rps), ("brier_severe", m_bs - f_bs)):
                s = np.bincount(groups, d, len(muni_codes))
                n = np.bincount(groups, minlength=len(muni_codes))
                boot = s[idx].sum(1) / n[idx].sum(1)
                row |= {f"d_{name}": d.mean(), f"d_{name}_ci_low": np.percentile(boot, 2.5),
                        f"d_{name}_ci_high": np.percentile(boot, 97.5)}
                if name == "rps":
                    row["municipalities_better"] = f"{int((s / n < 0).sum())}/{len(muni_codes)}"
            row["claim"] = ("better than fragility" if row["d_rps"] < 0 and row["d_rps_ci_high"] < 0
                            else "no clear difference" if row["d_rps_ci_low"] < 0 else "worse than fragility")
            row |= {"mean_p_severe": m_ps.mean(), "mean_p_severe_fragility": f_ps.mean(),
                    "observed_severe": (y >= C.SEVERE_MIN_GRADE).mean()}
            rows.append(row)
    head = pd.DataFrame(rows)
    head.to_csv(T / f"13_headline{sfx}.csv", index=False)
    print(("DRY RUN (development district standing in for the test district; not a result)\n" if mode != "final"
           else "") + HEADLINE_RULE + "\n")
    print(head.round(5).to_string(index=False))
    print("\nAll runs (test RPS; validation RPS where available). Selection was fixed before the test.")
    cols = ["run_id", "role", "model", "config", "feature_set", "pga_source", "pga_level", "rps", "brier_severe",
            "log_loss", "qwk", "macro_f1", "accuracy", "val_rps_municipality_folds", "val_rps_lodo"]
    print(allr.sort_values(["pga_source", "rps"])[cols].round(4).to_string(index=False))
    for col in ("val_rps_municipality_folds", "val_rps_lodo"):
        v = allr.dropna(subset=[col])
        print(f"Rank agreement, test vs {col}: Spearman {spearmanr(v[col], v.rps)[0]:.2f} over {len(v)} runs")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    for m in ("list", "freeze", "dry_run", "run", "report", "dry_report"):
        g.add_argument(f"--{m}", action="store_true")
    p.add_argument("--n_jobs", type=int, default=8)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="with --dry_run / --dry_report: 30k/10k rows")
    p.add_argument("--resume", action="store_true", help="with --run: complete an interrupted final test")
    a = p.parse_args()
    if a.list:
        d = run_list()
        d.to_csv(T / "13_freeze_list_DRAFT.csv", index=False)
        pd.set_option("display.width", 220)
        print(d.drop(columns="params").to_string(index=False))
        print(f"\n{len(d)} runs; {int(d.selected.sum())} validation-selected; by role:\n{d.role.value_counts().to_string()}")
    elif a.freeze:
        freeze()
    elif a.dry_run:
        run(a, "dry")
    elif a.run:
        run(a, "final")
    elif a.report:
        report("final", False)
    else:
        report("dry", a.quick)


if __name__ == "__main__":
    main()
