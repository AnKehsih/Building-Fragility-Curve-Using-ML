# Data sources and licences

This repository contains code, result tables, figures and a few small derived tables. It does not redistribute any raw dataset. The sources and their licences are listed below so that anyone reusing the derived files can meet the original terms.

| Data | Source | Licence | What is in this repository |
|---|---|---|---|
| Building survey (2015 Gorkha earthquake) | Household Registration for Housing Reconstruction Survey, Central Bureau of Statistics and National Planning Commission, Nepal; released through the 2015 Nepal Earthquake Open Data Portal (built with Kathmandu Living Labs). Copy used: Kaggle dataset "Earthquake Magnitude, Damage and Impact". | CC0 1.0 (public domain), as listed on the Kaggle dataset page when it was downloaded (October 2026) | No raw rows. Only aggregate results, the municipality list and the cross-validation fold assignment per municipality (`data/processed/`). |
| Ground shaking | U.S. Geological Survey ShakeMap, event `us20002926` (M7.8 Gorkha, 25 April 2015): Atlas re-run v1 (2020) and NEIC v9 (2015) | Public domain (U.S. Government work). USGS asks to be credited as the source. | No grid files. Mean PGA per municipality and per district (`data/processed/pga_*.csv`), computed by `code/05_link_pga_to_municipalities.py`. |
| Municipality boundaries | Open Data Nepal, "Administrative boundaries geographic data of 753 municipalities in Nepal" (author: Sagar Ghimire), downloaded 2 November 2026 | Creative Commons Attribution Share-Alike (CC BY-SA) | No boundary files. The boundaries were used only to average PGA within each municipality. |

## What this means for reuse

- **Survey (CC0):** no conditions. Citing the survey and the portal is good practice and is done in the paper.
- **USGS ShakeMap:** no conditions. Credit: U.S. Geological Survey, ShakeMap for event us20002926.
- **Boundaries (CC BY-SA):** attribution is required, and adapted material must be shared under the same licence. The files `data/processed/pga_municipality.csv` and `pga_district.csv` were produced with these boundaries. To stay on the safe side they are offered under **CC BY-SA 4.0**, with credit to Open Data Nepal / Sagar Ghimire for the boundaries and to USGS for the shaking values.
- **Everything else in `data/processed/` and `results/`** is derived from the CC0 survey and is released under CC0 1.0.

The household and individual tables of the survey are not used anywhere in this project, and no personal data is in this repository.

Download details (URLs, dates, versions and SHA-256 hashes of the downloaded files) are in `data/external/shakemap/SOURCE.md` and `data/external/boundaries/SOURCE.md`.

This file covers data only. No licence has been chosen for the code yet.
