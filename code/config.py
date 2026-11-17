"""
Central configuration for the ES 678 earthquake-damage project.

Purpose : single source of truth for paths, seeds, CV settings, column groups
          and the final-test district. Every numbered script imports this file.
Inputs  : none (constants only).
Outputs : none.
Usage   : `from config import *` (scripts in code/) or
          `import sys; sys.path.insert(0, "code"); import config`.

Rule: no hard-coded absolute paths anywhere else. The data root below is the
only place the raw-data location is stated; it is relative to the project root.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "NEPAL DATA"          # raw CSVs, read-only, never modified
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"    # ShakeMap, boundaries (+ source notes)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"  # cleaned tables, PGA table, splits
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

STRUCTURE_CSV = DATA_ROOT / "csv_building_structure.csv"
WARD_MAPPING_CSV = DATA_ROOT / "ward_vdcmun_district_name_mapping.csv"
# csv_building_damage_assessment.csv: descriptive checks only, never an input.
# household_* and individual_* tables: NOT used (privacy).

# ---------------------------------------------------------------- randomness
SEED = 678                      # master seed
CV_REPEAT_SEEDS = [678, 679, 680]   # alternative fold assignments (CONTEXT 5.2.2)
N_FOLDS = 5                     # grouped K-fold by municipality on development districts

# ---------------------------------------------------------------- target
TARGET = "damage_grade"          # stored as text, e.g. "Grade 3"
GRADES = [1, 2, 3, 4, 5]
N_GRADES = 5
SEVERE_MIN_GRADE = 4             # "severe damage" = grade >= 4 (Brier / reliability)

# ---------------------------------------------------------------- ids (grouping / linking only)
ID_COLS = ["building_id", "ward_id", "vdcmun_id", "district_id"]
GROUP_COL = "vdcmun_id"          # CV group = municipality
DISTRICTS = {12: "Okhaldhunga", 20: "Sindhuli", 21: "Ramechhap", 22: "Dolakha",
             23: "Sindhupalchok", 24: "Kavrepalanchok", 28: "Nuwakot",
             29: "Rasuwa", 30: "Dhading", 31: "Makwanpur", 36: "Gorkha"}

# Decision D5 (PROGRESS.md, 2026-11-02): Kavrepalanchok. Set ONCE, before any modelling;
# never look at its performance until all modelling choices are frozen.
FINAL_TEST_DISTRICT_ID = 24

# Decision D6. "municipality" planned; "district" = fallback / sensitivity test.
PGA_LEVEL = "municipality"

# ---------------------------------------------------------------- leakage (CONTEXT 3.3)
# Recorded AFTER the earthquake. Never model inputs; only the labelled
# leakage demonstration (code/12_failure_audit.py) may touch them.
POST_EQ_COLS = ["count_floors_post_eq", "height_ft_post_eq",
                "condition_post_eq", "technical_solution_proposed"]

# ---------------------------------------------------------------- allowed inputs (CONTEXT 3.4)
AGE_MISSING_FROM = 995           # age_building >= 995 is treated as missing
NUMERIC_FEATURES = ["count_floors_pre_eq", "age_building",
                    "plinth_area_sq_ft", "height_ft_pre_eq"]
CATEGORICAL_FEATURES = ["land_surface_condition", "position", "plan_configuration",
                        "foundation_type", "roof_type", "ground_floor_type",
                        "other_floor_type"]
SUPERSTRUCTURE_PREFIX = "has_superstructure_"   # 11 non-exclusive 0/1 flags (picked up by prefix)
SHAKING_FEATURE = "pga"          # added in step 05; log(PGA) used by fragility curves

# Decision D4: rule mapping superstructure flags to one structural type for the fragility
# baseline. Rules are defined in utils/preprocessing.py and compared in 03_structural_type.py.
STRUCTURAL_TYPE_RULE = "precedence_mixed"

# ---------------------------------------------------------------- external data (CONTEXT 3.5)
SHAKEMAP_EVENT_ID = "us20002926"   # M7.8 Gorkha, 25 April 2015
# Decision D7: primary PGA source = USGS-preferred ShakeMap (Atlas v4 re-run, 2020; the kind of
# automatic map issued today within hours of an event). NEIC v9 (2015) = sensitivity test only.
# Decision D11 (replaces the "sensitivity only" part of D7): both maps are reported side by side;
# configurations are selected separately per map and both go to the final test. The names below are
# kept; "PRIMARY" is now only the reference map for single-map extras (random-split comparison, θ plots).
SHAKEMAP_PRIMARY = "preferred"        # data/processed/shakemap_preferred.parquet
SHAKEMAP_SENSITIVITY = "neic_v9"      # data/processed/shakemap_neic_v9.parquet
SHAKEMAP_SOURCES = [SHAKEMAP_PRIMARY, SHAKEMAP_SENSITIVITY]
