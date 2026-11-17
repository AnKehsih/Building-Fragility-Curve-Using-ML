"""
02b_consistency_checks.py

Purpose : Data-integrity checks beyond 02_eda.py, on the 10 DEVELOPMENT districts only:
          (1) cross-field domain rules (utils.preprocessing.consistency_flags): records whose
              fields contradict each other or are physically implausible. Policy: flag, never
              delete; flags feed the failure audit (12_failure_audit.py).
          (2) association among categorical inputs (bias-corrected Cramer's V), to know which
              inputs carry the same information (shared credit in importance; redundancy).
          (3) variance inflation factors (VIF) of the ordinal-logit design matrix, to know which
              coefficients must not be interpreted individually.
          Input shift between development and test districts is NOT checked here: it uses the
          test district, so it is deferred to 13_final_test.py (inputs only, after freezing).
Inputs  : data/processed/buildings_clean.parquet (via utils.data)
Outputs : results/tables/02b_consistency_flags.csv     n, share, severe share flagged vs not, per rule
          results/tables/02b_cramers_v.csv              pairwise association of categorical inputs
          results/tables/02b_vif.csv                    VIF per design column
          results/tables/02b_console_output.txt         (redirect stdout, see Run)
Run     : .venv/Scripts/python code/02b_consistency_checks.py > results/tables/02b_console_output.txt
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.data import load_development
from utils.preprocessing import (assign_structural_type, consistency_flags, CONSISTENCY_RULES, P)

T = C.TABLES_DIR
df = load_development()
df["severe"] = (df[C.TARGET] >= C.SEVERE_MIN_GRADE).astype(int)
df["structural_type"] = assign_structural_type(df, C.STRUCTURAL_TYPE_RULE)
superstructure = [c for c in df.columns if c.startswith(P)]
print(f"Development data: {len(df):,} buildings")

# ------------------------------------------------------------------ 1. cross-field rules
flags = consistency_flags(df)
rows = []
for rule, (severity, desc) in CONSISTENCY_RULES.items():
    hit = flags[rule]
    rows.append({"rule": rule, "severity": severity, "description": desc,
                 "n": int(hit.sum()), "share": hit.mean(),
                 "share_severe_flagged": df.loc[hit, "severe"].mean() if hit.any() else np.nan,
                 "share_severe_not_flagged": df.loc[~hit, "severe"].mean()})
rules = pd.DataFrame(rows).round(4)
any_flag = flags.any(axis=1)
hard = flags[[r for r, (s, _) in CONSISTENCY_RULES.items() if s != "unusual"]].any(axis=1)
rules.loc[len(rules)] = {"rule": "any_contradiction_or_implausible", "severity": "summary",
                         "description": "c* or p* rule", "n": int(hard.sum()), "share": round(hard.mean(), 4),
                         "share_severe_flagged": round(df.loc[hard, "severe"].mean(), 4),
                         "share_severe_not_flagged": round(df.loc[~hard, "severe"].mean(), 4)}
rules.loc[len(rules)] = {"rule": "any_flag", "severity": "summary", "description": "any rule",
                         "n": int(any_flag.sum()), "share": round(any_flag.mean(), 4),
                         "share_severe_flagged": round(df.loc[any_flag, "severe"].mean(), 4),
                         "share_severe_not_flagged": round(df.loc[~any_flag, "severe"].mean(), 4)}
rules.to_csv(T / "02b_consistency_flags.csv", index=False)
print("\n1. Cross-field consistency rules (flag, never delete):\n",
      rules.drop(columns="description").to_string(index=False))

# Are flagged records concentrated in a few municipalities (a sign of survey-team practice)?
per_muni = hard.groupby(df["vdcmun_id"]).mean()
print(f"\nShare of buildings with a contradiction/implausible flag, per municipality: "
      f"median {per_muni.median():.4f}, max {per_muni.max():.4f} (municipality {per_muni.idxmax()})")


# ------------------------------------------------------------------ 2. categorical association
def cramers_v(x, y):
    """Bias-corrected Cramer's V (Bergsma 2013)."""
    table = pd.crosstab(x, y).values
    n = table.sum()
    chi2 = chi2_contingency(table, correction=False)[0]
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(kc - 1, rc - 1))))


cat_vars = C.CATEGORICAL_FEATURES + ["structural_type"]
storey_band = df["count_floors_pre_eq"].clip(upper=4).rename("storeys_band")
cv_df = pd.concat([df[cat_vars].astype(str), storey_band.astype(str)], axis=1)
names = list(cv_df.columns)
V = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
for a, b in combinations(names, 2):
    V.loc[a, b] = V.loc[b, a] = cramers_v(cv_df[a], cv_df[b])
V = V.round(3)
V.to_csv(T / "02b_cramers_v.csv")
print("\n2. Cramer's V among categorical inputs (1 = same information):\n", V.to_string())
pairs = [(a, b, V.loc[a, b]) for a, b in combinations(names, 2) if V.loc[a, b] >= 0.5]
print("Pairs with V >= 0.5:", sorted(pairs, key=lambda t: -t[2]) or "none")

# ------------------------------------------------------------------ 3. VIF of the ordinal-logit design
# Design as the ordinal logit will see it: numeric inputs (log plinth area), one-hot categoricals
# with the most frequent level dropped as reference, and the 0/1 superstructure flags.
# Missing values filled with the median / mode for this diagnostic only (no model is fitted).
X = df[C.NUMERIC_FEATURES].copy()
X["plinth_area_sq_ft"] = np.log(X["plinth_area_sq_ft"])
X = X.fillna(X.median())
for col in C.CATEGORICAL_FEATURES:
    s = df[col].fillna(df[col].mode()[0])
    dummies = pd.get_dummies(s, prefix=col, dtype=float)
    X = X.join(dummies.drop(columns=f"{col}_{s.value_counts().idxmax()}"))
X = X.join(df[superstructure].astype(float))
X = X.loc[:, X.std() > 0]
R = np.corrcoef(X.values, rowvar=False)
vif = pd.Series(np.diag(np.linalg.pinv(R)), index=X.columns, name="VIF").sort_values(ascending=False).round(2)
vif.to_csv(T / "02b_vif.csv")
print(f"\n3. VIF of the ordinal-logit design ({X.shape[1]} columns). VIF > 5 (coefficient not "
      f"interpretable on its own):\n", vif[vif > 5].to_string() if (vif > 5).any() else "none")
print("Top 10 VIF:\n", vif.head(10).to_string())
