# Supplementary material

Supplement to the paper *Predicting Earthquake Damage to Buildings from Construction Attributes and Ground Shaking* (ES 678, IIT Gandhinagar). The four-page paper stands on its own. This page points to the full results behind it, in the order of the paper's sections. Every file listed here is produced by a numbered script in `code/`.

Map names: the paper's **Atlas map** (2020 re-run) is `preferred` in file names and tables; the **NEIC map** (2015) is `neic_v9`.

## S1. Data and preparation (paper Sec. 2)

| What | File |
|---|---|
| The 31 checks of the raw data against the facts recorded beforehand | `results/tables/01_verification.csv` |
| Missing values, district summary | `results/tables/01_missing_values.csv`, `01_district_summary.csv` |
| Data-quality flags (contradictory or implausible records, kept) | `results/tables/02_quality_flags.csv`, `02b_consistency_flags.csv` |
| Grade shares by district; severe damage by storeys and age | `results/figures/02_grades_by_district.png`, `02_severe_vs_storeys_age.png` |
| Structural-type rule: the three rules compared, and grade shares by type | `results/tables/03_rule_comparison.csv`, `03_structural_types.csv`; `results/figures/03_grades_by_structural_type.png` |
| The two ShakeMaps compared; PGA per municipality and district | `results/tables/04_shakemap_pref_vs_neic.csv`; `data/processed/pga_municipality.csv`, `pga_district.csv` |
| Boundary name matching (110/110) | `results/tables/05_name_matching.csv` |
| PGA against observed damage | `results/figures/05b_pga_vs_damage.png` |

## S2. Baselines, models and selection (paper Secs. 3–4)

| What | File |
|---|---|
| Cross-validation folds (municipality to fold, 3 seeds) and their balance | `data/processed/folds_municipality.csv`; `results/tables/06_fold_balance.csv` |
| Fitted fragility curve parameters (median and dispersion per type) | `results/tables/07_fragility_params_full_dev.csv`, `07_fragility_params_cv.csv`; `results/figures/07_fragility_curves.png` |
| Every model and setting on every fold | `results/tables/08_cv_metrics_main.csv`, `09_cv_metrics_main.csv` |
| Selected settings and how much the selection flatters them | `results/tables/08b_selected.csv`, `09b_selected_final.csv`, `08b_selection_optimism.csv` |
| Random split against municipality folds | `results/tables/08b_random_vs_municipality.csv` |

## S3. Results and errors (paper Sec. 5)

| What | File |
|---|---|
| Fig. 1 values (gain over the fragility curves under four designs) | `results/tables/14_fig1_data.csv` |
| Final test: headline table with cluster-bootstrap intervals | `results/tables/13_headline.csv` |
| Final test: all 69 frozen runs, including secondary and post-hoc ones | `results/tables/13_all_runs.csv` |
| Freeze record (run list, claim rule, SHA-256 hashes) | `results/tables/13_freeze.json`, `13_freeze_list.csv` |
| Leave-one-district-out, per district | `results/tables/09c_summary.csv`, `09c_by_district.csv`, `09c_paired_vs_fragility.csv` |
| Errors by district, storeys, age, slope and structural type (incl. steep slopes) | `results/tables/12_breakdown.csv`; `results/figures/12_rps_by_subgroup.png` |

## S4. Graduate depth analysis (paper Sec. 6)

| What | File |
|---|---|
| 2×2 ablation (type + PGA only against all attributes, for every model) | `results/tables/09b_ablation.csv` |
| The four LightGBM fixes compared (PGA bins, district PGA, start from the fragility curve) | `results/tables/09b_lgbm_diagnosis.csv`, `09b_lgbm_variant_ranking.csv` |
| Early-stopping rounds, grouped against random folds | `results/tables/08b_lgbm_rounds.csv` |
| Overfitting: train–validation gap per model; capacity check | `results/tables/09c_summary.csv`, `09d_summary.csv` |
| Input sensitivity: drop-one-group refits and permutation, both models and maps | `results/tables/09e_importance.csv`, `09e_drop_summary.csv`, `09e_permutation_summary.csv` |
| Reduced input set (six groups dropped together) | `results/tables/09f_summary.csv` |
| Implied fragility curves for every structural type, both maps | `results/figures/11_severe_by_type_preferred.png`, `11_severe_by_type_neic_v9.png`; `results/tables/11_implied_params.csv` |
| Accumulated local effects (all inputs, and by storeys / structural type) | `results/figures/11b_ale.png`, `11b_ale_stratified.png`; `results/tables/11b_ale.csv` |
| Grouped SHAP contributions of the hybrid | `results/figures/11b_shap.png`; `results/tables/11b_shap_groups.csv` |
| Associations between inputs | `results/figures/11b_association.png` |
| Calibration: reliability diagram, Brier decomposition, by district | `results/figures/10_reliability.png`, `10_district_calibration.png`; `results/tables/10_calibration.csv` |
| Municipal against district PGA | `results/tables/10_resolution_sensitivity.csv` |
| Where the error sits (district / municipality / ward / within ward) | `results/tables/12_residual_decomposition.csv` |
| Twins and the label-noise floor | `results/tables/12_twins.csv` |
| Labelled leakage demonstration (post-earthquake fields) | `results/tables/12_leakage_demo_summary.csv` |

## S5. Reproducibility

- How to run everything, in order: `README.md`.
- Data sources and licences: `DATA_LICENSES.md`.
- AI-use disclosure: `README.md` and Section 8 of the paper.
