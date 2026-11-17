# Predicting Earthquake Damage to Buildings from Construction Attributes and Ground Shaking

Do machine-learning models that use pre-earthquake building attributes together with ground shaking predict damage-grade probabilities on unseen districts better than lognormal fragility curves? This repository holds the code, results and paper for that question, using the building survey of the Gorkha (Nepal) earthquake.

**Result.** On a held-out district, scored once after all modelling choices were fixed, a gradient-boosted model that starts from the fragility curve and learns a correction from building attributes improved the ranked probability score by 3% on both ShakeMaps. Tree models given shaking directly showed no reliable gain.

## Contents

| Path | What it holds |
|---|---|
| `paper/` | The paper (`main.tex`, `main.pdf`) with its figures and table |
| `code/` | Numbered scripts, run in order; shared modules in `code/utils/`; settings in `code/config.py` |
| `results/` | All result tables and figures |
| `data/processed/` | Small derived tables: cross-validation folds and shaking per municipality |
| `SUPPLEMENTARY.md` | Where each result behind the paper is |
| `DATA_LICENSES.md` | Data sources and licences |

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Download the survey CSV files (Kaggle: "Earthquake Magnitude, Damage and Impact") into a folder named `NEPAL DATA/` in the project root. The ShakeMaps and municipal boundaries are downloaded by the scripts.
3. Run the scripts in `code/` in numerical order from the project root. Each script's header lists its inputs, outputs and options.

## Data sources

- **Building survey:** Central Bureau of Statistics and National Planning Commission, Government of Nepal, *Household Registration for Housing Reconstruction Survey*, 2015 Nepal Earthquake Open Data Portal (with Kathmandu Living Labs). Kaggle copy "Earthquake Magnitude, Damage and Impact". CC0.
- **Ground shaking:** U.S. Geological Survey, *ShakeMap*, event `us20002926` (NEIC and Atlas versions). Public domain.
- **Municipal boundaries:** S. Ghimire, *Administrative boundaries of 753 municipalities in Nepal*, Open Data Nepal. CC BY-SA.

See `DATA_LICENSES.md` for details.

## AI use

Claude Code (Anthropic) was used to write and debug the code, run the analyses, design the figures and draft the text, at the author's direction. The author chose the question, data and baselines and made the key modelling decisions.
