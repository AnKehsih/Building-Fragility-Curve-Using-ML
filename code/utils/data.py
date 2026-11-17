"""
Shared data loaders. Every script that needs building data must load it through
these functions, so the final-test district cannot be read by accident.

  load_development()  -> the 10 development districts (all EDA, CV, tuning)
  load_final_test()   -> the held-out district (config.FINAL_TEST_DISTRICT_ID);
                         refuses to run unless models are declared frozen.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

CLEAN_PARQUET = C.PROCESSED_DIR / "buildings_clean.parquet"


def _load_all():
    return pd.read_parquet(CLEAN_PARQUET)


def load_development():
    """Cleaned buildings outside the final-test district."""
    if C.FINAL_TEST_DISTRICT_ID is None:
        raise RuntimeError("config.FINAL_TEST_DISTRICT_ID is not set (Decision D5).")
    df = _load_all()
    dev = df[df["district_id"] != C.FINAL_TEST_DISTRICT_ID].reset_index(drop=True)
    assert C.FINAL_TEST_DISTRICT_ID not in set(dev["district_id"])
    return dev


def attach_pga(df, source=None, level="municipality"):
    """Return a copy of df with a 'pga' column (g) from ShakeMap `source` (default
    config.SHAKEMAP_PRIMARY) averaged at `level` ('municipality' or 'district').
    Deterministic lookup from 05_link_pga_to_municipalities.py; nothing is fitted."""
    source = source or C.SHAKEMAP_PRIMARY
    if level == "municipality":
        table, key = pd.read_csv(C.PROCESSED_DIR / "pga_municipality.csv"), "vdcmun_id"
    elif level == "district":
        table, key = pd.read_csv(C.PROCESSED_DIR / "pga_district.csv"), "district_id"
    else:
        raise ValueError(level)
    out = df.copy()
    out["pga"] = out[key].map(table.set_index(key)[f"pga_{source}"])
    if out["pga"].isna().any():
        raise ValueError(f"{int(out['pga'].isna().sum())} buildings without PGA ({source}, {level})")
    return out


def load_posteq_for_leakage_demo():
    """Post-earthquake columns (CONTEXT 3.3) of the DEVELOPMENT buildings, keyed by building_id. Only for
    the clearly labelled leakage demonstration in 12_failure_audit.py; never a model input anywhere else."""
    p = pd.read_parquet(C.PROCESSED_DIR / "buildings_posteq.parquet")
    return p[p["building_id"].isin(load_development()["building_id"])].reset_index(drop=True)


def load_final_test(models_frozen=False):
    """Cleaned buildings of the final-test district. Use ONCE, in 13_final_test.py,
    after every modelling choice is frozen and logged in PROGRESS.md."""
    if not models_frozen:
        raise PermissionError("Final-test district is locked. Pass models_frozen=True only "
                              "in 13_final_test.py after logging the freeze in PROGRESS.md.")
    df = _load_all()
    return df[df["district_id"] == C.FINAL_TEST_DISTRICT_ID].reset_index(drop=True)
