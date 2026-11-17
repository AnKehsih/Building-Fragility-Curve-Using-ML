"""
Deterministic (not learned) preprocessing shared across scripts.

assign_structural_type(): maps the 11 non-exclusive has_superstructure_* flags to ONE
structural type for the fragility baseline. It uses only the building's own flags, so it
can be applied to any row without fitting. Three candidate rules (Decision D4, compared in
03_structural_type.py):

  "precedence"        strongest load-bearing system wins (CONTEXT 5.1 preliminary rule)
  "precedence_mixed"  as "precedence", but a building that combines a strong system
                      (RC or cement-mortar masonry) with a weak, mud-based or dry masonry
                      gets its own class "mixed_strong_weak"
  "weakest_first"     most vulnerable load-bearing material wins
"""
import pandas as pd

P = "has_superstructure_"

STRONG = ["rc_engineered", "rc_non_engineered", "cement_mortar_brick", "cement_mortar_stone"]
WEAK = ["mud_mortar_stone", "stone_flag", "mud_mortar_brick", "adobe_mud"]
LOAD_BEARING = STRONG + WEAK

# (type name, flags that qualify), highest precedence first. Timber and bamboo are usually
# secondary members (floors, roof, bands) in masonry buildings, so they only define a type
# when no load-bearing masonry/RC flag is present.
PRECEDENCE = [
    ("rc_engineered", ["rc_engineered"]),
    ("rc_non_engineered", ["rc_non_engineered"]),
    ("cement_mortar_masonry", ["cement_mortar_brick", "cement_mortar_stone"]),
    ("mud_mortar_brick", ["mud_mortar_brick"]),
    ("stone_masonry", ["mud_mortar_stone", "stone_flag"]),
    ("adobe_mud", ["adobe_mud"]),
    ("timber_bamboo", ["timber", "bamboo"]),
    ("other", ["other"]),
]
# Most vulnerable first: rubble/field-stone and adobe (EMS-98 class A), then mud-mortar
# brick, cement-mortar masonry, non-engineered RC, engineered RC.
WEAKEST_FIRST = [
    ("stone_masonry", ["mud_mortar_stone", "stone_flag"]),
    ("adobe_mud", ["adobe_mud"]),
    ("mud_mortar_brick", ["mud_mortar_brick"]),
    ("cement_mortar_masonry", ["cement_mortar_brick", "cement_mortar_stone"]),
    ("rc_non_engineered", ["rc_non_engineered"]),
    ("rc_engineered", ["rc_engineered"]),
    ("timber_bamboo", ["timber", "bamboo"]),
    ("other", ["other"]),
]
MIXED = "mixed_strong_weak"
RULES = ["precedence", "precedence_mixed", "weakest_first"]


def _any(df, flags):
    return df[[P + f for f in flags]].sum(axis=1) > 0


def _apply(df, ordered_rules):
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, flags in ordered_rules:
        out = out.mask(out.isna() & _any(df, flags), name)
    return out


def type_order(rule):
    if rule == "precedence_mixed":
        return [n for n, _ in PRECEDENCE[:4]] + [MIXED] + [n for n, _ in PRECEDENCE[4:]]
    return [n for n, _ in (WEAKEST_FIRST if rule == "weakest_first" else PRECEDENCE)]


def assign_structural_type(df, rule="precedence_mixed"):
    """Return a categorical Series with one structural type per building."""
    if rule == "precedence":
        out = _apply(df, PRECEDENCE)
    elif rule == "weakest_first":
        out = _apply(df, WEAKEST_FIRST)
    elif rule == "precedence_mixed":
        out = _apply(df, PRECEDENCE).mask(_any(df, STRONG) & _any(df, WEAK), MIXED)
    else:
        raise ValueError(f"unknown rule {rule!r}; choose from {RULES}")
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} buildings matched no structural-type rule")
    return pd.Categorical(out, categories=type_order(rule))


CONSISTENCY_RULES = {
    # name: (severity, description). "contradiction" = two fields cannot both be true;
    # "implausible" = physically very unlikely; "unusual" = possible but worth auditing.
    "c1_one_storey_has_upper_floor": ("contradiction", "1 storey but other_floor_type is not 'Not applicable'"),
    "c2_multi_storey_no_upper_floor": ("contradiction", ">1 storey but other_floor_type is 'Not applicable'"),
    "p1_storey_height_lt_5ft": ("implausible", "height_ft_pre_eq / storeys < 5 ft"),
    "p2_storey_height_gt_20ft": ("implausible", "height_ft_pre_eq / storeys > 20 ft"),
    "p3_plinth_lt_100sqft": ("implausible", "plinth area < 100 sq ft (about 9 m2)"),
    "u1_rc_eng_without_rc_elements": ("unusual", "engineered RC flag, but no RC foundation, RC ground floor or RCC roof"),
    "u2_rc_on_mud_foundation": ("unusual", "RC flag with a mud-mortar stone/brick foundation"),
    "u3_rcc_roof_on_weak_masonry_only": ("unusual", "RCC roof on a building whose only load-bearing material is weak (mud/dry/adobe)"),
    "u4_rc_ground_floor_no_strong_system": ("unusual", "RC ground floor but no RC or cement-mortar flag"),
}


def consistency_flags(df):
    """Deterministic cross-field checks (no fitting). Returns one boolean column per rule.
    Policy: flag, never delete; flags feed the failure audit."""
    storeys = df["count_floors_pre_eq"]
    per_storey = df["height_ft_pre_eq"] / storeys
    upper_na = df["other_floor_type"] == "Not applicable"
    any_rc = _any(df, ["rc_engineered", "rc_non_engineered"])
    strong, weak = _any(df, STRONG), _any(df, WEAK)
    f = pd.DataFrame(index=df.index)
    f["c1_one_storey_has_upper_floor"] = (storeys == 1) & ~upper_na
    f["c2_multi_storey_no_upper_floor"] = (storeys > 1) & upper_na
    f["p1_storey_height_lt_5ft"] = per_storey < 5
    f["p2_storey_height_gt_20ft"] = per_storey > 20
    f["p3_plinth_lt_100sqft"] = df["plinth_area_sq_ft"] < 100
    f["u1_rc_eng_without_rc_elements"] = (df[P + "rc_engineered"] == 1) & (df["foundation_type"] != "RC") \
        & (df["ground_floor_type"] != "RC") & (df["roof_type"] != "RCC/RB/RBC")
    f["u2_rc_on_mud_foundation"] = any_rc & (df["foundation_type"] == "Mud mortar-Stone/Brick")
    f["u3_rcc_roof_on_weak_masonry_only"] = (df["roof_type"] == "RCC/RB/RBC") & weak & ~strong
    f["u4_rc_ground_floor_no_strong_system"] = (df["ground_floor_type"] == "RC") & ~strong
    return f


def n_load_bearing_materials(df):
    """Number of distinct load-bearing material flags (masonry/RC), ignoring timber/bamboo/other."""
    return df[[P + f for f in LOAD_BEARING]].sum(axis=1)
