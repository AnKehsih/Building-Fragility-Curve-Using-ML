"""
05_link_pga_to_municipalities.py

Purpose : Give every municipality (vdcmun_id) and district a PGA value.
          1. Download Nepal local-level (municipality) boundaries from Open Data Nepal (CKAN API).
          2. Match survey municipality names (ward_vdcmun_district_name_mapping.csv) to boundary
             names, within the same district; report every unmatched municipality.
          3. Mean PGA of ShakeMap grid points inside each municipality polygon, for the primary
             and sensitivity ShakeMaps (config.SHAKEMAP_PRIMARY / SHAKEMAP_SENSITIVITY).
          4. District-level mean PGA (fallback / sensitivity, Decision D6).
Inputs  : data/processed/municipalities.csv (from 01), data/processed/shakemap_<label>.parquet (from 04),
          Open Data Nepal dataset "nepal-municipalities-wise-geographic-data-shp-geojson-topojson-kml"
Outputs : data/external/boundaries/                 raw zip + extracted shapefile (never edited), SOURCE.md
          data/processed/pga_municipality.csv       vdcmun_id, names, n grid points, mean PGA per map
          data/processed/pga_district.csv           district_id, mean PGA per map
          results/tables/05_name_matching.csv       survey name -> boundary name, match method
          results/tables/05_console_output.txt      (redirect stdout, see Run)
Run     : .venv/Scripts/python -I code/05_link_pga_to_municipalities.py > results/tables/05_console_output.txt
"""
import difflib
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import requests
import shapefile  # pyshp
import shapely
from shapely.geometry import shape
from shapely.ops import transform, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.EXTERNAL_DIR / "boundaries"
OUT.mkdir(parents=True, exist_ok=True)
CKAN = "https://api.opendatanepal.com/api/3/action/package_show"
DATASET = "nepal-municipalities-wise-geographic-data-shp-geojson-topojson-kml"

# ------------------------------------------------------------------ 1. download
pkg = requests.get(CKAN, params={"id": DATASET}, timeout=60).json()["result"]
res = next(r for r in pkg["resources"] if r["format"].upper() == "SHAPE")
zpath = OUT / "nepal_municipalities_shp.zip"
if not zpath.exists():
    r = requests.get(res["url"], timeout=600)
    r.raise_for_status()
    zpath.write_bytes(r.content)
raw = zpath.read_bytes()
assert len(raw) == int(res["size"]), f"size {len(raw)} != listed {res['size']}"
meta = {"dataset": DATASET, "title": pkg["title"], "license": pkg.get("license_title"),
        "author": pkg.get("author"), "resource_url": res["url"], "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest()}
(OUT / "boundaries_metadata.json").write_text(json.dumps(meta, indent=2))

with zipfile.ZipFile(zpath) as z:
    members = z.namelist()
    for m in members:
        if not (OUT / "shp" / m).exists():
            z.extract(m, OUT / "shp")
shp = next((OUT / "shp").rglob("*.shp"))
prj = shp.with_suffix(".prj")
print("Zip members:", members)
print("CRS (.prj):", prj.read_text()[:300] if prj.exists() else "MISSING")

sf = shapefile.Reader(str(shp))
fields = [f[0] for f in sf.fields[1:]]
print(f"Shapes: {len(sf)}  type: {sf.shapeTypeName}  bbox: {[round(v, 3) for v in sf.bbox]}")
print("Fields:", fields)
for rec in sf.records()[:3]:
    print("  ", dict(zip(fields, rec)))

# ------------------------------------------------------------------ 2. reproject to WGS84 lon/lat
# The .prj gives Transverse Mercator (lon_0 84, k 0.9999, x_0 500000) on the Everest 1830 (1937
# adjustment) ellipsoid but labels the datum "unknown". That is Nepal's national grid on the
# Nepal 1981 datum (EPSG:6207), so the EPSG "Nepal 1981 to WGS 84 (1)" Helmert shift is applied.
# This datum is an inference from the ellipsoid and projection; the size of the shift is reported.
TM = "+proj=tmerc +lat_0=0 +lon_0=84 +k=0.9999 +x_0=500000 +y_0=0 +ellps=evrst30 +units=m +no_defs"
to_wgs84 = pyproj.Transformer.from_crs(TM + " +towgs84=293.17,726.18,245.36,0,0,0,0", "EPSG:4326",
                                       always_xy=True).transform
to_wgs84_noshift = pyproj.Transformer.from_crs(TM + " +towgs84=0,0,0", "EPSG:4326", always_xy=True).transform

recs = pd.DataFrame(sf.records(), columns=fields)
geoms = [transform(to_wgs84, shape(s.__geo_interface__)) for s in sf.shapes()]
c_shift = [g.centroid for g in geoms]
c_plain = [transform(to_wgs84_noshift, shape(s.__geo_interface__)).centroid for s in sf.shapes()[:50]]
geod = pyproj.Geod(ellps="WGS84")
shift_m = [geod.inv(a.x, a.y, b.x, b.y)[2] for a, b in zip(c_shift[:50], c_plain)]
print(f"Datum shift moves polygon centroids by {np.mean(shift_m):.0f} m on average "
      f"(ShakeMap grid spacing about 1,800 m)")
print("Unit types in boundary file:", recs["UNIT_TYPE"].value_counts().to_dict())


# ------------------------------------------------------------------ 3. match survey names to boundary names
def norm(name):
    s = name.lower()
    # "metropolit(i)an": the survey spells Hetauda "Sub-Metropolitian City"
    s = re.sub(r"\b(rural municipality|sub-metropolit(i)?an city|metropolit(i)?an city|municipality|"
               r"gaunpalika|nagarpalika)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


# Spelling differences in district names (boundary file -> survey spelling).
DISTRICT_ALIASES = {"kabhrepalanchok": "kavrepalanchok", "makawanpur": "makwanpur"}
# Local levels renamed after the boundary file was made. Each target is the ONLY boundary polygon
# in that district left unclaimed by exact/fuzzy matching (checked on 2026-11-02, and asserted below).
MANUAL_NAMES = {  # (survey district, survey name normalised) -> boundary UNIT_NAME
    ("Ramechhap", "likhutamakoshi"): "Likhu",
    ("Dolakha", "baitedhar"): "Baiteshwor",
    ("Nuwakot", "myagang"): "Meghang",
    ("Rasuwa", "aamachhodingmo"): "Parbati Kunda",
    ("Dhading", "netrawatidabajong"): "Netrawati",
    ("Gorkha", "barpaksulikot"): "Sulikot",
    ("Gorkha", "bhimsenthapa"): "Bhimsen",
}

recs["district_norm"] = recs["DISTRICT"].map(norm).replace(DISTRICT_ALIASES)
recs["name_norm"] = recs["UNIT_NAME"].map(norm)
recs["geom"] = geoms
muni = pd.read_csv(C.PROCESSED_DIR / "municipalities.csv")
muni["district_norm"] = muni["district_name"].map(norm)
muni["name_norm"] = muni["vdcmun_name"].map(norm)

# district names must match exactly after normalisation (11 districts; checked here)
missing_d = sorted(set(muni["district_norm"]) - set(recs["district_norm"]))
print("Survey districts not found in boundary file:", missing_d or "none")

rows = []
for _, m in muni.iterrows():
    cand = recs[recs["district_norm"] == m["district_norm"]]
    exact = cand[cand["name_norm"] == m["name_norm"]]
    manual = MANUAL_NAMES.get((m["district_name"], m["name_norm"]))
    if len(exact):
        hit, method, score = exact, "exact", 1.0
    elif manual:
        hit, method, score = cand[cand["UNIT_NAME"] == manual], "manual_renamed", np.nan
        assert len(hit), f"manual target {manual} not found in {m['district_name']}"
    else:
        scores = cand["name_norm"].map(lambda n: difflib.SequenceMatcher(None, m["name_norm"], n).ratio())
        best = scores.max() if len(scores) else 0.0
        hit = cand[scores == best] if best >= 0.75 else cand.iloc[0:0]
        method, score = ("fuzzy" if len(hit) else "unmatched"), round(float(best), 3)
    rows.append({"vdcmun_id": m["vdcmun_id"], "vdcmun_name": m["vdcmun_name"],
                 "district_id": m["district_id"], "district_name": m["district_name"],
                 "boundary_name": "; ".join(sorted(set(hit["UNIT_NAME"]))),
                 "boundary_type": "; ".join(sorted(set(hit["UNIT_TYPE"]))),
                 "n_polygons": len(hit), "method": method, "score": score,
                 "boundary_index": list(hit.index)})
match = pd.DataFrame(rows)
# a boundary polygon must not be claimed by two survey municipalities
claimed = match.explode("boundary_index").dropna(subset=["boundary_index"])
dupes = claimed[claimed["boundary_index"].duplicated(keep=False)]
match.drop(columns="boundary_index").to_csv(C.TABLES_DIR / "05_name_matching.csv", index=False)
print("\nMatching summary:", match["method"].value_counts().to_dict())
print("Fuzzy matches (check by eye):\n",
      match.loc[match["method"] == "fuzzy", ["district_name", "vdcmun_name", "boundary_name", "score"]].to_string(index=False)
      if (match["method"] == "fuzzy").any() else "none")
print("Unmatched:\n", match.loc[match["method"] == "unmatched", ["district_name", "vdcmun_name", "score"]].to_string(index=False)
      if (match["method"] == "unmatched").any() else "none")
print("Manual (renamed) matches:\n",
      match.loc[match["method"] == "manual_renamed", ["district_name", "vdcmun_name", "boundary_name"]].to_string(index=False))
print("Boundary polygons claimed twice:", "none" if dupes.empty else dupes[["vdcmun_name", "boundary_index"]].to_string())
assert dupes.empty, "a boundary polygon is matched to two survey municipalities"
# every polygon of a surveyed district (excluding parks/reserves) should be used exactly once
LOCAL = {"Gaunpalika", "Nagarpalika", "Upamahanagarpalika", "Mahanagarpalika"}
surveyed = recs[recs["district_norm"].isin(set(muni["district_norm"])) & recs["UNIT_TYPE"].isin(LOCAL)]
unused = surveyed.loc[~surveyed.index.isin(claimed["boundary_index"].astype(int)), ["DISTRICT", "UNIT_NAME"]]
print("Local-level polygons in surveyed districts not used:", "none" if unused.empty else unused.to_string(index=False))

# ------------------------------------------------------------------ 4. mean PGA inside each municipality
maps = {lbl: pd.read_parquet(C.PROCESSED_DIR / f"shakemap_{lbl}.parquet")
        for lbl in (C.SHAKEMAP_PRIMARY, C.SHAKEMAP_SENSITIVITY)}


def mean_pga(geom, grid):
    """Mean PGA (g) of grid points inside geom; nearest grid point to the centroid if none fall inside."""
    xmin, ymin, xmax, ymax = geom.bounds
    g = grid[grid["lon"].between(xmin, xmax) & grid["lat"].between(ymin, ymax)]
    inside = g[shapely.contains_xy(geom, g["lon"].values, g["lat"].values)]
    if len(inside):
        return inside["pga_g"].mean(), len(inside)
    c = geom.centroid
    i = ((grid["lon"] - c.x) ** 2 + (grid["lat"] - c.y) ** 2).idxmin()
    return grid.at[i, "pga_g"], 0


out = []
for _, m in match[match["method"] != "unmatched"].iterrows():
    geom = unary_union([recs.at[i, "geom"] for i in m["boundary_index"]])
    row = {"vdcmun_id": m["vdcmun_id"], "vdcmun_name": m["vdcmun_name"], "district_id": m["district_id"],
           "district_name": m["district_name"], "area_km2": abs(geod.geometry_area_perimeter(geom)[0]) / 1e6,
           "lon_centroid": geom.centroid.x, "lat_centroid": geom.centroid.y}
    for lbl, grid in maps.items():
        row[f"pga_{lbl}"], row[f"n_points_{lbl}"] = mean_pga(geom, grid)
    out.append(row)
pga_m = pd.DataFrame(out).round({"area_km2": 1, "lon_centroid": 4, "lat_centroid": 4,
                                 f"pga_{C.SHAKEMAP_PRIMARY}": 4, f"pga_{C.SHAKEMAP_SENSITIVITY}": 4})
pga_m.to_csv(C.PROCESSED_DIR / "pga_municipality.csv", index=False)

# ------------------------------------------------------------------ 5. district mean PGA (area-based: all grid points in the district)
dist_rows = []
for did, grp in match[match["method"] != "unmatched"].groupby("district_id"):
    dname = grp["district_name"].iloc[0]
    geom = unary_union(recs.loc[recs["district_norm"] == norm(dname), "geom"].tolist())
    row = {"district_id": did, "district_name": dname}
    for lbl, grid in maps.items():
        row[f"pga_{lbl}"], row[f"n_points_{lbl}"] = mean_pga(geom, grid)
    dist_rows.append(row)
pga_d = pd.DataFrame(dist_rows).round(4)
pga_d.to_csv(C.PROCESSED_DIR / "pga_district.csv", index=False)

p, s = f"pga_{C.SHAKEMAP_PRIMARY}", f"pga_{C.SHAKEMAP_SENSITIVITY}"
print(f"\nMunicipalities with PGA: {len(pga_m)} of {len(muni)}; grid points per municipality: "
      f"min {pga_m[f'n_points_{C.SHAKEMAP_PRIMARY}'].min()}, median {int(pga_m[f'n_points_{C.SHAKEMAP_PRIMARY}'].median())}")
print(f"Municipal PGA ({C.SHAKEMAP_PRIMARY}): min {pga_m[p].min():.3f} g, median {pga_m[p].median():.3f} g, max {pga_m[p].max():.3f} g")
print(f"Municipal PGA ({C.SHAKEMAP_SENSITIVITY}): min {pga_m[s].min():.3f} g, median {pga_m[s].median():.3f} g, max {pga_m[s].max():.3f} g")
print(f"Spearman between maps across municipalities: {pga_m[p].corr(pga_m[s], method='spearman'):.3f}")
within = pga_m.groupby("district_name")[p].agg(["min", "max", "count"]).round(3)
within["max/min"] = (within["max"] / within["min"]).round(2)
print("\nWithin-district range of municipal PGA (primary map):\n", within.to_string())
print("\nDistrict mean PGA:\n", pga_d.to_string(index=False))

(OUT / "SOURCE.md").write_text(
    "# Nepal local-level (municipality) boundaries\n\n"
    f"Downloaded 2026-11-02 by `code/05_link_pga_to_municipalities.py` from Open Data Nepal (CKAN dataset `{DATASET}`).\n"
    f"Title: {pkg['title']}  \nAuthor: {pkg.get('author')}  \nLicence: {pkg.get('license_title')} (attribution required; share-alike).\n"
    f"Resource URL, size and SHA-256: `boundaries_metadata.json`.\n\n"
    "CRS: Transverse Mercator (lon_0 84, k 0.9999, x_0 500000) on Everest 1830 (1937); datum labelled 'unknown' in the .prj; "
    "treated as Nepal 1981 (EPSG:6207) with the EPSG Helmert shift to WGS 84. Files are not edited.\n")
