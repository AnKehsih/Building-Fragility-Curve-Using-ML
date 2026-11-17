"""
04_shakemap_download_parse.py

Purpose : Download the USGS ShakeMap ground-motion grid for the 2015 Gorkha earthquake
          (event us20002926) and convert it to a plain table of grid points.
          Main source  = the USGS-preferred ShakeMap product (first in the event's product list;
                         the Atlas re-run, ShakeMap v4, processed 2020-06-03).
          Alternative  = the original NEIC ShakeMap (source "us", version 9, 2015-07-02),
                         kept only for a PGA-source sensitivity check.
          Product versions, URLs, timestamps and SHA-256 hashes are written to a metadata file.
Inputs  : USGS FDSN event API (internet), https://earthquake.usgs.gov/fdsnws/event/1/query
Outputs : data/external/shakemap/<label>_grid.xml(.zip)   raw downloads (never edited)
          data/external/shakemap/shakemap_metadata.json    versions, URLs, hashes, grid spec
          data/processed/shakemap_<label>.parquet          lon, lat, pga_g, pgv_cms, mmi, ...
          results/tables/04_shakemap_summary.csv
Run     : .venv/Scripts/python code/04_shakemap_download_parse.py     (from the project root)
          Re-running skips files that are already downloaded.

Units: ShakeMap grid PGA is in %g; it is converted to g (pga_g = PGA / 100).
"""
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

OUT = C.EXTERNAL_DIR / "shakemap"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# label -> how to pick the product from the event's "shakemap" list
PRODUCTS = {
    "preferred": dict(pick=lambda ps: ps[0], file="download/grid.xml"),
    "neic_v9": dict(pick=lambda ps: next(p for p in ps if p["source"] == "us"),
                    file="download/grid.xml.zip"),
}
# Approximate study-area box (11 surveyed districts), used only for the summary table.
STUDY_BOX = dict(lon_min=84.0, lon_max=87.0, lat_min=27.0, lat_max=28.8)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def get_bytes(url, dest):
    if dest.exists():
        return dest.read_bytes()
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return r.content


def parse_grid_xml(raw):
    """Parse a ShakeMap grid.xml into a DataFrame plus its header attributes."""
    root = ET.fromstring(raw)
    ns = {"sm": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    tag = (lambda t: f"sm:{t}") if ns else (lambda t: t)
    spec = root.find(tag("grid_specification"), ns).attrib
    fields = sorted(root.findall(tag("grid_field"), ns), key=lambda f: int(f.attrib["index"]))
    names = [f.attrib["name"].lower() for f in fields]
    units = {f.attrib["name"].lower(): f.attrib.get("units", "") for f in fields}
    values = np.array(root.find(tag("grid_data"), ns).text.split(), dtype=float)
    grid = pd.DataFrame(values.reshape(-1, len(names)), columns=names)
    expected = int(spec["nlon"]) * int(spec["nlat"])
    assert len(grid) == expected, f"grid has {len(grid)} rows, header says {expected}"
    header = {"event": root.find(tag("event"), ns).attrib, "shakemap_grid": root.attrib,
              "grid_specification": spec, "fields": names, "units": units}
    return grid, header


event = requests.get(API, params={"eventid": C.SHAKEMAP_EVENT_ID, "format": "geojson"}, timeout=60)
event.raise_for_status()
shakemaps = event.json()["properties"]["products"]["shakemap"]

meta, summaries = {}, []
for label, spec in PRODUCTS.items():
    prod = spec["pick"](shakemaps)
    content = prod["contents"][spec["file"]]
    fname = f"{label}_{Path(spec['file']).name}"
    raw = get_bytes(content["url"], OUT / fname)
    assert len(raw) == content["length"], f"{fname}: size {len(raw)} != listed {content['length']}"
    xml_bytes = raw
    if fname.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            xml_bytes = z.read(next(n for n in z.namelist() if n.endswith(".xml")))

    grid, header = parse_grid_xml(xml_bytes)
    if header["units"].get("pga") not in ("pctg", "%g"):     # ShakeMap v3 writes "pctg", v4 "%g"
        raise ValueError(f"unexpected PGA units: {header['units'].get('pga')}")
    grid = grid.rename(columns={"pgv": "pgv_cms"})
    grid["pga_g"] = grid["pga"] / 100.0
    grid.to_parquet(C.PROCESSED_DIR / f"shakemap_{label}.parquet", index=False)

    meta[label] = {
        "product_source": prod["source"], "product_code": prod["code"],
        "product_version": prod["properties"].get("version"),
        "map_status": prod["properties"].get("map-status"),
        "process_timestamp": prod["properties"].get("process-timestamp"),
        "update_time_ms": prod["updateTime"], "url": content["url"], "file": fname,
        "bytes": len(raw), "sha256": sha256(raw), **header,
    }
    box = grid[grid["lon"].between(STUDY_BOX["lon_min"], STUDY_BOX["lon_max"])
               & grid["lat"].between(STUDY_BOX["lat_min"], STUDY_BOX["lat_max"])]
    summaries.append({
        "label": label, "source": prod["source"], "version": prod["properties"].get("version"),
        "processed": prod["properties"].get("process-timestamp"),
        "grid_points": len(grid), "dlon": header["grid_specification"]["nominal_lon_spacing"],
        "dlat": header["grid_specification"]["nominal_lat_spacing"],
        "pga_g_max_all": grid["pga_g"].max(), "box_points": len(box),
        "box_pga_g_min": box["pga_g"].min(), "box_pga_g_median": box["pga_g"].median(),
        "box_pga_g_max": box["pga_g"].max(),
    })

# ------------------------------------------------------------------ anchor check against a recording
# KATNP (Kathmandu, USGS/NSC strong-motion station, approx. 27.712 N, 85.316 E). Its recorded
# horizontal PGA is remembered as about 0.16 g (Takai et al. 2016, Earth Planets Space 68:10).
# UNVERIFIED: both maps give ~0.32-0.37 g there, so either the value or the station's role in
# the maps needs checking against the paper before this anchor is used for any decision.
# Kathmandu is outside the 11 survey districts; this only checks each map's level.
KATNP = (85.316, 27.712)
anchor = []
for label in PRODUCTS:
    g = pd.read_parquet(C.PROCESSED_DIR / f"shakemap_{label}.parquet")
    i = ((g["lon"] - KATNP[0]) ** 2 + (g["lat"] - KATNP[1]) ** 2).idxmin()
    row = {"label": label, "grid_lon": g.at[i, "lon"], "grid_lat": g.at[i, "lat"],
           "pga_g": g.at[i, "pga_g"], "mmi": g.at[i, "mmi"], "recorded_pga_g_approx": 0.16}
    if "stdpga" in g:
        row["stdpga_ln"] = g.at[i, "stdpga"]
    anchor.append(row)
anchor = pd.DataFrame(anchor).round(3)
anchor.to_csv(C.TABLES_DIR / "04_shakemap_katnp_check.csv", index=False)
print("\nGrid value nearest KATNP (Kathmandu) vs recorded ~0.16 g:\n", anchor.to_string(index=False))

# ------------------------------------------------------------------ preferred vs NEIC v9 over the study box
# Nearest NEIC grid point for every preferred grid point inside the box (spacings are ~1 arc-min).
a =pd.read_parquet(C.PROCESSED_DIR / "shakemap_preferred.parquet")
b = pd.read_parquet(C.PROCESSED_DIR / "shakemap_neic_v9.parquet")
a = a[a["lon"].between(STUDY_BOX["lon_min"], STUDY_BOX["lon_max"])
      & a["lat"].between(STUDY_BOX["lat_min"], STUDY_BOX["lat_max"])].reset_index(drop=True)
dist, idx = cKDTree(b[["lon", "lat"]].values).query(a[["lon", "lat"]].values)
ratio = a["pga_g"].values / b["pga_g"].values[idx]
compare = pd.Series({
    "points": len(a), "max_match_distance_deg": dist.max(),
    "pga_ratio_pref_over_neic_median": np.median(ratio),
    "pga_ratio_q25": np.percentile(ratio, 25), "pga_ratio_q75": np.percentile(ratio, 75),
    "spearman_pga": pd.Series(a["pga_g"].values).corr(pd.Series(b["pga_g"].values[idx]), method="spearman"),
    "mmi_median_pref": a["mmi"].median(), "mmi_median_neic": np.median(b["mmi"].values[idx]),
}).round(4)
compare.to_csv(C.TABLES_DIR / "04_shakemap_pref_vs_neic.csv", header=["value"])
print("\nPreferred vs NEIC v9 over the study box:\n", compare.to_string())

meta["retrieved_with"] = f"{API}?eventid={C.SHAKEMAP_EVENT_ID}&format=geojson"
(OUT / "shakemap_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
summary = pd.DataFrame(summaries).round(4)
summary.to_csv(C.TABLES_DIR / "04_shakemap_summary.csv", index=False)
print(summary.T.to_string())
for label in PRODUCTS:
    print(f"\n{label}: fields {meta[label]['fields']}; units {meta[label]['units']}")
    print(f"  event: {meta[label]['event']}")
