"""
01_load_and_clean.py

Purpose : Load the raw building survey, re-verify the facts recorded in CONTEXT.md
          Section 3.2 (counts, missing values, age code 999, leakage example),
          and write a cleaned table. Raw files are only read, never modified.
Inputs  : NEPAL DATA/csv_building_structure.csv
          NEPAL DATA/ward_vdcmun_district_name_mapping.csv
Outputs : data/processed/buildings_clean.parquet      pre-earthquake inputs + ids + damage grade
          data/processed/buildings_posteq.parquet     post-earthquake columns, SEPARATE FILE on purpose:
                                                      only the labelled leakage demonstration may read it
          data/processed/municipalities.csv           vdcmun_id, names, district (for ShakeMap linking)
          results/tables/01_verification.csv          expected vs observed for each CONTEXT fact
          results/tables/01_district_summary.csv      n, municipalities, grade shares per district
          results/tables/01_missing_values.csv        missing counts per column
Run     : .venv/Scripts/python code/01_load_and_clean.py     (from the project root)

Cleaning rules (all fixed here, none learned from data):
  - drop rows with missing damage_grade (no label);
  - damage_grade "Grade k" -> integer k;
  - age_building >= config.AGE_MISSING_FROM (995) -> NaN (999 is a placeholder code);
  - other missing values are left as NaN and imputed later INSIDE training folds.
No scaling, encoding or imputation is done here, so nothing is learned from the data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
C.TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ load
raw = pd.read_csv(C.STRUCTURE_CSV)
mapping = pd.read_csv(C.WARD_MAPPING_CSV)
for df in (raw, mapping):
    for col in df.columns[df.dtypes == "object"].tolist() + df.select_dtypes("string").columns.tolist():
        df[col] = df[col].str.strip()

checks = []   # (fact, expected, observed)


def check(fact, expected, observed):
    checks.append({"fact": fact, "expected": expected, "observed": observed,
                   "match": expected == observed})


# ------------------------------------------------------------------ re-verify CONTEXT 3.2
check("rows", 762106, len(raw))
check("columns", 31, raw.shape[1])
check("districts", 11, raw["district_id"].nunique())
check("municipalities (vdcmun_id)", 110, raw["vdcmun_id"].nunique())
check("wards", 945, raw["ward_id"].nunique())
check("duplicate building_id", 0, int(raw["building_id"].duplicated().sum()))

grade_counts = raw[C.TARGET].value_counts(dropna=False)
for k, expected in zip(C.GRADES, [78815, 87257, 136412, 183844, 275766]):
    check(f"Grade {k} count", expected, int(grade_counts.get(f"Grade {k}", 0)))
check("damage_grade missing", 12, int(raw[C.TARGET].isna().sum()))
check("technical_solution_proposed missing", 12, int(raw["technical_solution_proposed"].isna().sum()))
check("position missing", 1, int(raw["position"].isna().sum()))
check("plan_configuration missing", 1, int(raw["plan_configuration"].isna().sum()))
# CONTEXT.md figures below were computed on the 762,094 rows that have a grade; one of the
# 12 ungraded rows also has age 999 and zero post-earthquake storeys (checked 2026-11-02).
graded = raw[raw[C.TARGET].notna()]
check("age_building == 999 (graded rows)", 3144, int((graded["age_building"] == 999).sum()))
check("age_building == 999 (all rows, incl. 1 ungraded)", 3145, int((raw["age_building"] == 999).sum()))
check("age_building max", 999, int(raw["age_building"].max()))
check("age_building >= 995 (rule, graded rows)", 3144,
      int((graded["age_building"] >= C.AGE_MISSING_FROM).sum()))

# Leakage evidence (CONTEXT 3.3): zero storeys after the earthquake => collapsed => Grade 5
zero_post = graded[graded["count_floors_post_eq"] == 0]
check("graded buildings with count_floors_post_eq == 0", 261353, len(zero_post))
check("... of which Grade 5", 261353, int((zero_post[C.TARGET] == "Grade 5").sum()))

# ------------------------------------------------------------------ ward mapping integrity
check("mapping: duplicate ward_id", 0, int(mapping["ward_id"].duplicated().sum()))
merged = raw[["building_id", "ward_id", "vdcmun_id", "district_id"]].merge(
    mapping, on="ward_id", how="left", suffixes=("", "_map"), validate="m:1")
check("buildings whose ward is missing from mapping", 0, int(merged["vdcmun_name"].isna().sum()))
check("vdcmun_id disagrees with mapping", 0, int((merged["vdcmun_id"] != merged["vdcmun_id_map"]).sum()))
check("district_id disagrees with mapping", 0, int((merged["district_id"] != merged["district_id_map"]).sum()))
check("district ids match config.DISTRICTS", sorted(C.DISTRICTS), sorted(raw["district_id"].unique().tolist()))

# ------------------------------------------------------------------ clean
df = raw.copy()
n_before = len(df)
df = df[df[C.TARGET].notna()].copy()
df[C.TARGET] = df[C.TARGET].str.extract(r"(\d)")[0].astype(int)
assert set(df[C.TARGET].unique()) == set(C.GRADES)
n_age_set_nan = int((df["age_building"] >= C.AGE_MISSING_FROM).sum())
df.loc[df["age_building"] >= C.AGE_MISSING_FROM, "age_building"] = np.nan
check("rows after dropping missing grade", 762094, len(df))

superstructure = [c for c in df.columns if c.startswith(C.SUPERSTRUCTURE_PREFIX)]
check("number of has_superstructure_* flags", 11, len(superstructure))

# Post-earthquake columns go to their own file so they cannot slip into a model by accident.
posteq = df[["building_id"] + C.POST_EQ_COLS].copy()
inputs = df.drop(columns=C.POST_EQ_COLS)
expected_cols = set(C.ID_COLS + [C.TARGET] + C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + superstructure)
unused = set(inputs.columns) - expected_cols
check("columns kept that are not in config feature lists", [], sorted(unused))

inputs.to_parquet(C.PROCESSED_DIR / "buildings_clean.parquet", index=False)
posteq.to_parquet(C.PROCESSED_DIR / "buildings_posteq.parquet", index=False)
muni = (mapping[["vdcmun_id", "vdcmun_name", "district_id", "district_name"]]
        .drop_duplicates().sort_values("vdcmun_id"))
muni = muni[muni["vdcmun_id"].isin(inputs["vdcmun_id"].unique())]
check("municipality rows in municipalities.csv", 110, len(muni))
check("vdcmun_id unique in municipalities.csv", True, bool(muni["vdcmun_id"].is_unique))
muni.to_csv(C.PROCESSED_DIR / "municipalities.csv", index=False)

# ------------------------------------------------------------------ tables
pd.DataFrame(checks).to_csv(C.TABLES_DIR / "01_verification.csv", index=False)

shares = (pd.crosstab(inputs["district_id"], inputs[C.TARGET], normalize="index")
          .rename(columns=lambda k: f"share_grade{k}"))
summary = pd.DataFrame({
    "district": pd.Series(C.DISTRICTS),
    "n_buildings": inputs.groupby("district_id").size(),
    "n_municipalities": inputs.groupby("district_id")["vdcmun_id"].nunique(),
}).join(shares).round(3)
summary["share_severe_g4plus"] = (inputs[inputs[C.TARGET] >= C.SEVERE_MIN_GRADE]
                                  .groupby("district_id").size() / summary["n_buildings"]).round(3)
summary.sort_values("share_grade5").to_csv(C.TABLES_DIR / "01_district_summary.csv")

missing = pd.DataFrame({"missing_raw": raw.isna().sum(), "missing_clean": inputs.isna().sum()})
missing["missing_clean"] = missing["missing_clean"].fillna(0).astype(int)
missing.to_csv(C.TABLES_DIR / "01_missing_values.csv")

# ------------------------------------------------------------------ report
res = pd.DataFrame(checks)
print(res.to_string(index=False))
print(f"\n{int(res['match'].sum())}/{len(res)} checks match CONTEXT.md")
print(f"\nrows: {n_before} raw -> {len(df)} clean; age set to NaN in {n_age_set_nan} rows")
print("\nDistrict summary (sorted by Grade 5 share):")
print(summary.sort_values("share_grade5").to_string())
print("\nMissing values in cleaned input table:")
print(missing[missing["missing_clean"] > 0].to_string())
if not res["match"].all():
    print("\nMISMATCHES:")
    print(res[~res["match"]].to_string(index=False))
