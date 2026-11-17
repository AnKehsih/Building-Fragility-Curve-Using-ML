"""
Model definitions shared by 08 (main CV), 09 (ablation), 10 (sensitivity) and 13 (final test).

Feature sets
  full      all allowed pre-earthquake inputs (CONTEXT 3.4) + structural_type + ln_pga
  type_pga  structural_type + ln_pga only (same information as the fragility baseline)

Every learned step (imputer medians, scaler means/SDs, one-hot levels, rare-level merging,
tree splits, coefficients) is fitted inside `fit` on the training fold only.

Hyperparameter grids are small and fixed in advance (GRIDS). Each configuration is scored on
the same development folds; the configuration with the lowest mean RPS on municipality folds is
selected afterwards (08b). The final-test district gives the unbiased score of the selected one.
Class weighting: none. Weighting changes the fitted probabilities (L10 notes), and the
primary metrics (RPS, Brier, calibration) judge probabilities.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils.ordinal import OrdinalLogit

SUPERSTRUCTURE = ["has_superstructure_adobe_mud", "has_superstructure_mud_mortar_stone",
                  "has_superstructure_stone_flag", "has_superstructure_cement_mortar_stone",
                  "has_superstructure_mud_mortar_brick", "has_superstructure_cement_mortar_brick",
                  "has_superstructure_timber", "has_superstructure_bamboo",
                  "has_superstructure_rc_non_engineered", "has_superstructure_rc_engineered",
                  "has_superstructure_other"]
CATEGORICAL = C.CATEGORICAL_FEATURES + ["structural_type"]
NUMERIC = C.NUMERIC_FEATURES + ["ln_pga"]
FEATURE_SETS = {
    "full": {"numeric": NUMERIC, "categorical": CATEGORICAL, "binary": SUPERSTRUCTURE},
    "type_pga": {"numeric": ["ln_pga"], "categorical": ["structural_type"], "binary": []},
    "no_pga": {"numeric": C.NUMERIC_FEATURES, "categorical": CATEGORICAL, "binary": SUPERSTRUCTURE},
    # LEAKAGE DEMONSTRATION ONLY (12_failure_audit.py, CONTEXT 3.3): full inputs plus the post-earthquake
    # columns. Data from utils.cvrun.model_data do not contain these columns, so any other use fails.
    "LEAKAGE_DEMO_posteq": {"numeric": NUMERIC + ["count_floors_post_eq", "height_ft_post_eq"],
                            "categorical": CATEGORICAL + ["condition_post_eq", "technical_solution_proposed"],
                            "binary": SUPERSTRUCTURE},
}
# Input groups for importance (09e). A group is permuted or dropped as a whole. "pga" holds both columns
# because the fragility curve (and so the hybrid's starting score) reads `pga`, the other models `ln_pga`.
# The 11 superstructure flags are one group: they are non-exclusive and strongly related.
FEATURE_GROUPS = {"pga": ["ln_pga", "pga"], "structural_type": ["structural_type"], "superstructure": SUPERSTRUCTURE} \
    | {c: [c] for c in C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES}
DROP = "-drop-"                # feature set "<base>-drop-<group>": the base set without that group (09e)
# Reduced input set (09f, post-hoc): without the six groups that 09e found negligible for both selected
# models on both maps (Step 25). "<base>_reduced" is the base set without them.
REDUCED_DROP = ["ground_floor_type", "height_ft_pre_eq", "land_surface_condition", "plan_configuration",
                "plinth_area_sq_ft", "position"]
_gone = {c for g in REDUCED_DROP for c in FEATURE_GROUPS[g]}
FEATURE_SETS |= {f"{b}_reduced": {k: [c for c in v if c not in _gone] for k, v in FEATURE_SETS[b].items()}
                 for b in ("full", "no_pga")}

GRIDS = {
    "ordinal_logit": [{"alpha": a} for a in (1.0, 100.0, 10000.0)],
    "decision_tree": [{"max_depth": d} for d in (3, 4, 5)],
    "random_forest": [{"min_samples_leaf": m} for m in (20, 100)],
    "lightgbm": [{"num_leaves": nl, "min_child_samples": mc} for nl in (31, 127) for mc in (100, 500)],
}
LGBM_FIXED = dict(objective="multiclass", learning_rate=0.05, n_estimators=3000, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
# LightGBM variants of 09 (diagnosis of the PGA-as-municipality-identifier problem, Step 18):
# (label, extra params, feature set, PGA level). "district_pga" is reported but excluded from
# selection (D13: on municipality folds district PGA identifies the district, which a new district cannot use).
LGBM_VARIANTS = [
    ("district_pga", {}, "full", "district"),
    ("pga_bins5", {"pga_bins": 5}, "full", "municipality"),
    ("pga_bins10", {"pga_bins": 10}, "full", "municipality"),
    ("init_fragility", {"init": "fragility"}, "no_pga", "municipality"),
]
EXCLUDED_FROM_SELECTION = {"district_pga": "D13"}
EARLY_STOPPING_ROUNDS = 100
INNER_VAL_SHARE = 0.15         # share of training municipalities held out for LightGBM early stopping


def config_name(params):
    return "_".join(f"{k}{v:g}" if isinstance(v, float) else f"{k}{v}" for k, v in params.items())


def _cols(fs):
    base, _, drop = fs.partition(DROP)
    s, gone = FEATURE_SETS[base], set(FEATURE_GROUPS[drop]) if drop else set()
    return tuple([c for c in s[k] if c not in gone] for k in ("numeric", "categorical", "binary"))


def linear_preprocessor(fs):
    num, cat, binary = _cols(fs)
    num_steps = []
    for col in num:
        steps = [SimpleImputer(strategy="median", add_indicator=(col == "age_building"))]
        if col == "plinth_area_sq_ft":
            steps.append(FunctionTransformer(np.log, feature_names_out="one-to-one"))
        steps.append(StandardScaler())
        num_steps.append((col, make_pipeline(*steps), [col]))
    cat_step = ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                     OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.001,
                                                   sparse_output=False)), cat)
    parts = num_steps + [cat_step] + ([("bin", "passthrough", binary)] if binary else [])
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def tree_preprocessor(fs):
    num, cat, binary = _cols(fs)
    parts = [("num", SimpleImputer(strategy="median", add_indicator=True), num),
             ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                   OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat)]
    if binary:
        parts.append(("bin", "passthrough", binary))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def _lgbm_frame(df, fs, categories=None):
    num, cat, binary = _cols(fs)
    X = df[num + cat + binary].copy()
    for c in cat:
        X[c] = pd.Categorical(X[c].astype(str), categories=None if categories is None else categories[c])
    return X


class LGBMWithEarlyStopping:
    """LightGBM multiclass; rounds chosen by early stopping on an inner validation split of the
    TRAINING fold (whole municipalities for grouped CV), then predictions on the outer fold.

    Extra params used by 09 (diagnosis), popped before LightGBM sees them:
      pga_bins=k        ln_pga gets at most k histogram bins (LightGBM bins by row counts), so trees
                        cannot split PGA finely enough to single out municipalities.
      init="fragility"  boost from the fragility curve: a FragilityModel (structural type + PGA) is
                        fitted on the inner-training rows and its log-probabilities are the starting
                        scores; the trees then only learn corrections from the feature set's columns.
      n_rounds=n        (09d) fixed number of boosting rounds, no early stopping; fitted on all
                        training rows (no inner validation split is needed).
    """

    def __init__(self, fs, params, threads, seed):
        self.fs, self.threads, self.seed = fs, threads, seed
        self.params = dict(params)
        self.pga_bins, self.init = self.params.pop("pga_bins", None), self.params.pop("init", None)
        self.n_rounds = self.params.pop("n_rounds", None)

    def _init_score(self, df):
        return np.log(np.clip(self.frag_.predict_proba(df), 1e-12, 1))

    def fit(self, df, y, groups=None):
        import lightgbm as lgb
        rng = np.random.default_rng(self.seed)
        if self.n_rounds:
            inner = np.zeros(len(df), bool)
        elif groups is not None:
            ug = np.unique(groups)
            inner = np.isin(groups, rng.choice(ug, max(1, int(round(INNER_VAL_SHARE * len(ug)))), replace=False))
        else:
            inner = rng.random(len(df)) < INNER_VAL_SHARE
        Xtr = _lgbm_frame(df.iloc[~inner], self.fs)
        self.categories_ = {c: Xtr[c].cat.categories for c in _cols(self.fs)[1]}
        kw = LGBM_FIXED | self.params
        if self.pga_bins:
            kw["max_bin_by_feature"] = [self.pga_bins if c == "ln_pga" else 255 for c in Xtr.columns]
        if self.n_rounds:
            kw["n_estimators"] = self.n_rounds
        self.model_ = lgb.LGBMClassifier(**kw, n_jobs=self.threads, random_state=self.seed)
        # Labels passed as 0..K-1: LightGBM 4.7 does not label-encode eval_y, so grades 1..5 would make
        # the early-stopping loss meaningless (found in the 2026-11-02 smoke test: stopped at ~20 rounds).
        y0 = np.asarray(y, int) - 1
        fit_kw = {}
        if not self.n_rounds:
            fit_kw = dict(eval_X=(_lgbm_frame(df.iloc[inner], self.fs, self.categories_),), eval_y=(y0[inner],),
                          eval_metric="multi_logloss",
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        if self.init == "fragility":
            from utils.fragility import FragilityModel
            self.frag_ = FragilityModel().fit(df.iloc[~inner])          # inner-training rows only
            fit_kw["init_score"] = self._init_score(df.iloc[~inner])
            if not self.n_rounds:
                fit_kw["eval_init_score"] = [self._init_score(df.iloc[inner])]
        self.model_.fit(Xtr, y0[~inner], **fit_kw)
        self.best_iteration_ = self.n_rounds or int(self.model_.best_iteration_ or LGBM_FIXED["n_estimators"])
        return self

    def predict_proba(self, df):
        X = _lgbm_frame(df, self.fs, self.categories_)
        if self.init == "fragility":                                     # predict_proba ignores init scores
            raw = self.model_.predict_proba(X, raw_score=True, num_iteration=self.best_iteration_)
            raw = raw + self._init_score(df)
            p = np.exp(raw - raw.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
        else:
            p = self.model_.predict_proba(X, num_iteration=self.best_iteration_)
        P = np.zeros((len(df), C.N_GRADES))
        P[:, np.asarray(self.model_.classes_, int)] = p
        return P


class SklearnModel:
    def __init__(self, pre, est):
        self.pre, self.est = pre, est

    def fit(self, df, y, groups=None):
        self.est.fit(self.pre.fit_transform(df), y)
        return self

    def predict_proba(self, df):
        P = np.zeros((len(df), C.N_GRADES))
        P[:, np.asarray(self.est.classes_, int) - 1] = self.est.predict_proba(self.pre.transform(df))
        return P


class BaselineModel:
    """Fragility curves (type + PGA, utils/fragility.py) or majority, with the model interface, so that
    baselines can run as CV tasks next to the ML models (09c). Feature set is ignored."""

    def __init__(self, name):
        self.name = name

    def fit(self, df, y, groups=None):
        from utils.fragility import FragilityModel, MajorityModel
        self.m_ = FragilityModel().fit(df) if self.name == "fragility" else MajorityModel().fit(np.asarray(y))
        return self

    def predict_proba(self, df):
        return self.m_.predict_proba(df) if self.name == "fragility" else self.m_.predict_proba(len(df))


def make_model(name, params, fs, threads, seed, quick=False, oob=False):
    if name in ("fragility", "majority"):
        return BaselineModel(name)
    if name == "ordinal_logit":
        return SklearnModel(linear_preprocessor(fs), OrdinalLogit(**params))
    if name == "decision_tree":
        return SklearnModel(tree_preprocessor(fs), DecisionTreeClassifier(min_samples_leaf=500, random_state=seed, **params))
    if name == "random_forest":
        return SklearnModel(tree_preprocessor(fs), RandomForestClassifier(
            n_estimators=20 if quick else 300, max_features="sqrt", oob_score=oob, n_jobs=threads,
            random_state=seed, **params))
    if name == "lightgbm":
        return LGBMWithEarlyStopping(fs, params, threads, seed)
    raise ValueError(name)


def _norm(P):
    P = np.clip(P, 1e-12, 1)
    return P / P.sum(1, keepdims=True)


def fit_predict(name, params, fs, train, val, y_train, threads, seed, groups=None, quick=False, train_rows=None):
    """Fit on the training rows only, return (val probabilities, info dict).
    train_rows: positions in `train` to predict as well (info["P_train"]), for the train-validation gap.
    Random forest: each tree saw ~63% of the training rows, so predicting training rows with all trees
    is optimistic by construction. P_train is then the out-of-bag prediction (only trees that did not see
    the row), and the all-trees prediction is kept as info["P_train_inbag"] for comparison."""
    t = time.perf_counter()
    oob = name == "random_forest" and train_rows is not None
    m = make_model(name, params, fs, threads, seed, quick, oob=oob).fit(train, y_train, groups)
    P = _norm(m.predict_proba(val))
    info = {"fit_seconds": round(time.perf_counter() - t, 1),
            "best_iteration": getattr(m, "best_iteration_", np.nan),
            "converged": getattr(getattr(m, "est", None), "converged_", np.nan)}
    if train_rows is not None:
        info["P_train"] = _norm(m.predict_proba(train.iloc[train_rows]))
    if oob:
        info["P_train_inbag"] = info["P_train"]
        P_oob = np.zeros((len(train_rows), C.N_GRADES))
        P_oob[:, np.asarray(m.est.classes_, int) - 1] = np.nan_to_num(m.est.oob_decision_function_[train_rows])
        info["P_train"] = _norm(P_oob)
    return P, info
