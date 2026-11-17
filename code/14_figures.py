"""
14_figures.py

Purpose : The paper's figures and main table, from saved results only (nothing is fitted; the test district
          is not read: its numbers come from the 13 report tables). Sized for the supplied neurips_2026.sty,
          whose text block is ONE column 5.5 in wide (the template README says "two-column", the style file
          does not); fonts are 7-8 pt at print size. White background (the paper is printed); in Fig. 1 the
          four designs differ by marker shape as well as shade, so they can be told apart in greyscale.
            Fig. 1  Gain over the fragility curves under four evaluation designs (random split, municipality
                    folds, leave-one-district-out, final test), every validation-selected model and the plain
                    LightGBM, both maps. Shows the random-split optimism, LODO's realism and the test result.
            Fig. 2  Interpretation and the district-wide error (preferred map): (a) P(grade >= 4) against PGA for
                    stone masonry (partial dependence, 11, with binned observed shares); (b) the hybrid's ALE
                    of age by storeys (11b); (c) each district's mean predicted vs observed severe share (10,
                    LODO), with Kavrepalanchok from the final test (13). (Merged from two figures to fit four
                    pages; reliability is reported in the text.)
            Table 1 Final-test metrics (LaTeX, booktabs): fragility curves, the four selected models and the
                    majority baseline, both maps, with the paired RPS difference and its 95% interval.
Inputs  : results/tables/07_cv_metrics.csv, 08/09_cv_metrics_main.csv, 08b_selected.csv, 09b_selected_final.csv,
          09c_lodo_metrics_main.csv, 10_calibration_by_district.csv, 11_curves.csv,
          11_observed.csv, 11b_ale_stratified.csv, 13_headline.csv, 13_all_runs.csv
Outputs : results/figures/14_fig{1,2}_*.pdf and .png, results/tables/14_table1_final_test.tex,
          14_fig1_data.csv; copies of the PDFs and the table in paper/figures and paper/tables
Run     : python code/14_figures.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils.plotting import CATEGORICAL, GRADE_COLORS, INK, MUTED, set_style

T, F = C.TABLES_DIR, C.FIGURES_DIR
PAPER = C.PROJECT_ROOT / "paper"
TEXT_WIDTH_IN = 5.5
MAPS = {"preferred": "Atlas ShakeMap (2020)", "neic_v9": "NEIC ShakeMap (2015)"}   # names used in the paper
LABEL = {"fragility": "Fragility curves", "ordinal_logit": "Ordinal logit", "decision_tree": "Decision tree",
         "random_forest": "Random forest", "lightgbm": "Hybrid LightGBM", "lightgbm_plain": "Plain LightGBM",
         "majority": "Majority class"}
MODEL_COLOR = {"fragility": CATEGORICAL[0], "ordinal_logit": CATEGORICAL[1], "lightgbm": CATEGORICAL[2],
               "lightgbm_plain": CATEGORICAL[3]}                      # same slots as Figs. 10-12
DESIGNS = ["Random split", "Municipality folds", "Leave-one-district-out", "Final test (Kavrepalanchok)"]
DESIGN_MARKER = ["o", "s", "^", "D"]
BG = "white"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(F / f"{name}.{ext}")
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(F / f"{name}.pdf", PAPER / "figures" / f"{name}.pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


def specs():
    """{(map, label): (model, config, feature_set)} for the selected models and the plain LightGBM."""
    s9 = pd.read_csv(T / "09b_selected_final.csv")
    s8 = pd.read_csv(T / "08b_selected.csv")
    out = {(r.pga_source, r.model): (r.model, r.config, r.feature_set) for r in s9.itertuples()}
    for src in MAPS:
        cfg = s8[(s8.scheme == "municipality") & (s8.pga_source == src) & (s8.model == "lightgbm")].config.iloc[0]
        out[src, "lightgbm_plain"] = ("lightgbm", cfg, "full")
    return out


# ---------------------------------------------------------------- Fig. 1
def gains():
    """Paired RPS difference vs the fragility curves (same map, municipality PGA) per design."""
    m89 = pd.concat([pd.read_csv(T / "08_cv_metrics_main.csv"), pd.read_csv(T / "09_cv_metrics_main.csv")])
    m89 = m89[m89.pga_level == "municipality"]
    c9 = pd.read_csv(T / "09c_lodo_metrics_main.csv")
    c9 = c9[c9.pga_level == "municipality"]
    b7 = pd.read_csv(T / "07_cv_metrics.csv")
    head = pd.read_csv(T / "13_headline.csv")
    allr = pd.read_csv(T / "13_all_runs.csv")
    rows = []
    for (src, label), (model, cfg, fs) in specs().items():
        pick = lambda d: d[(d.model == model) & (d.config == cfg) & (d.feature_set == fs) & (d.pga_source == src)]
        for design, scheme, base, frag in (
                (DESIGNS[0], "random", m89, b7[b7.model == f"fragility_{src}_municipality"]),
                (DESIGNS[1], "municipality", m89, b7[b7.model == f"fragility_{src}_municipality"]),
                (DESIGNS[2], "district", c9, c9[(c9.model == "fragility") & (c9.pga_source == src)])):
            m = pick(base)
            m = m[m.scheme == scheme].drop_duplicates(["seed", "fold"])
            f = frag[frag.scheme == scheme].drop_duplicates(["seed", "fold"])
            p = m[["seed", "fold", "rps"]].merge(f[["seed", "fold", "rps"]], on=["seed", "fold"], suffixes=("", "_f"))
            if len(p) < 2:
                continue
            d = p.rps - p.rps_f
            half = 1.96 * d.std() / np.sqrt(len(d))
            rows.append({"pga_source": src, "model": label, "design": design, "d": d.mean(), "lo": d.mean() - half,
                         "hi": d.mean() + half, "n": len(d), "interval": "mean +/- 1.96 SE over folds"})
        h = head[(head.pga_source == src) & (head.model == model) & (head.config == cfg) & (head.feature_set == fs)]
        if len(h) and label != "lightgbm_plain":
            r = h.iloc[0]
            rows.append({"pga_source": src, "model": label, "design": DESIGNS[3], "d": r.d_rps, "lo": r.d_rps_ci_low,
                         "hi": r.d_rps_ci_high, "n": 13, "interval": "95% cluster bootstrap, 13 municipalities"})
        else:
            a = pick(allr)
            a = a[a.pga_level == "municipality"]
            fr = allr[(allr.model == "fragility") & (allr.pga_source == src) & (allr.pga_level == "municipality")]
            rows.append({"pga_source": src, "model": label, "design": DESIGNS[3], "d": a.rps.iloc[0] - fr.rps.iloc[0],
                         "lo": np.nan, "hi": np.nan, "n": 13, "interval": "secondary run: point estimate only"})
    return pd.DataFrame(rows)


def fig1(g):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    order = ["lightgbm", "ordinal_logit", "random_forest", "decision_tree", "lightgbm_plain"]
    ramp = [GRADE_COLORS[0], GRADE_COLORS[1], GRADE_COLORS[3], INK]    # ordered by realism: light -> dark
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 1.7), sharey=True)
    for ax, src in zip(axes, MAPS):
        ax.axvline(0, color=MUTED, lw=0.8)
        for j, des in enumerate(DESIGNS):
            d = g[(g.pga_source == src) & (g.design == des)].set_index("model").reindex(order)
            y = np.arange(len(order)) + (j - 1.5) * 0.17
            ok = d.d.notna()
            ax.errorbar(d.d[ok], y[ok], xerr=[(d.d - d.lo)[ok].fillna(0), (d.hi - d.d)[ok].fillna(0)],
                        fmt=DESIGN_MARKER[j], ms=3.5, color=ramp[j], ecolor=ramp[j], elinewidth=1, capsize=0,
                        label=des, zorder=3, markeredgecolor=BG, markeredgewidth=0.5)
        ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
        ax.set_title(MAPS[src])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()                                  # shared y: invert once (twice would undo it)
    fig.supxlabel("RPS minus fragility curves (negative = better)", y=0.14, fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "14_fig1_gain_by_design")


# ---------------------------------------------------------------- Fig. 2
def fig2():
    """(a) implied P(grade >= 4) for stone masonry (11), (b) ALE of age by storeys (11b), (c) district severe share:
    LODO for the development districts (10), Kavrepalanchok from the final test (13). Preferred map."""
    import matplotlib.pyplot as plt
    src = C.SHAKEMAP_PRIMARY
    cu, ob = pd.read_csv(T / "11_curves.csv"), pd.read_csv(T / "11_observed.csv")
    st = pd.read_csv(T / "11b_ale_stratified.csv", dtype={"stratum": str})
    bd = pd.read_csv(T / "10_calibration_by_district.csv")
    head = pd.read_csv(T / "13_headline.csv")
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.25]})

    ax = axes[0]
    for m in ["fragility", "ordinal_logit", "lightgbm", "lightgbm_plain"]:
        g = cu[(cu.pga_source == src) & (cu.structural_type == "stone_masonry") & (cu.model == m) & (cu.k == 4)]
        ax.plot(g.pga, g.p, color=MODEL_COLOR[m], lw=1.3, ls="--" if m == "fragility" else "-", label=LABEL[m])
    o = ob[(ob.pga_source == src) & (ob.structural_type == "stone_masonry") & (ob.k == 4)]
    ax.scatter(o.pga, o.p, marker="D", s=10, color=INK, zorder=4, label="Observed")
    ax.set(ylim=(0, 1), xlabel="PGA (g)", ylabel="P(grade ≥ 4)", title="(a) Stone masonry")

    ax = axes[1]
    s = st[(st.pga_source == src) & (st.model == "lightgbm") & (st.feature == "age_building")]
    for j, k in enumerate(["1", "2", "3"]):                  # storeys are ordered: one-hue ramp
        g = s[s.stratum == k]
        ax.plot(g.x, 100 * g.ale, color=GRADE_COLORS[2 * j], lw=1.3, label=f"{k} storey" + ("s" if k != "1" else ""))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xlabel="Age (years)", ylabel="Δ P(grade ≥ 4), points", title="(b) Age effect (hybrid)")
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[2]
    models = ["fragility", "ordinal_logit", "lightgbm"]
    d = bd[(bd.pga_source == src) & (bd.pga_level == "municipality")]
    obs = d.groupby("district")["observed"].first()
    rows = obs.sort_values().index.tolist()
    obs = obs.to_dict()
    pred = {(r.district, r.model): r.mean_p for r in d.itertuples()}
    h = head[head.pga_source == src].set_index("model")
    test = "Kavrepalanchok"
    obs[test] = h.observed_severe.iloc[0]
    pred[test, "fragility"] = h.mean_p_severe_fragility.iloc[0]
    for m in ("ordinal_logit", "lightgbm"):
        pred[test, m] = h.loc[m, "mean_p_severe"]
    rows = [test] + rows
    for i, m in enumerate(models):
        ax.scatter([pred[r, m] for r in rows], np.arange(len(rows)) + (i - 1) * 0.2, s=10, color=MODEL_COLOR[m],
                   edgecolor=BG, lw=0.4, zorder=3)
    ax.scatter([obs[r] for r in rows], range(len(rows)), marker="D", s=12, color=INK, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(rows)), rows, fontsize=6)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set(xlim=(0, 1), xlabel="Share with grade ≥ 4", title="(c) District severe share")
    ax.grid(axis="y", visible=False)

    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "14_fig2_interpretation")


# ---------------------------------------------------------------- Table 1
def table1():
    allr = pd.read_csv(T / "13_all_runs.csv")
    head = pd.read_csv(T / "13_headline.csv")
    order = ["majority", "fragility", "decision_tree", "random_forest", "ordinal_logit", "lightgbm"]
    shown = allr[(allr.selected) | ((allr.model == "fragility") & (allr.pga_level == "municipality"))]
    best_rps = shown.groupby("pga_source").rps.min().to_dict()
    lines = [r"\begin{tabular}{@{}llrrrrrl@{}}", r"\toprule",
             r"Map & Model & RPS & Brier$_{\geq 4}$ & Log loss & QWK & Acc. & $\Delta$RPS vs fragility [95\% CI] \\",
             r"\midrule"]
    for src in MAPS:
        for m in order:
            if m == "majority":
                if src != C.SHAKEMAP_PRIMARY:
                    continue
                r = allr[allr.model == "majority"].iloc[0]
            elif m == "fragility":
                r = allr[(allr.model == m) & (allr.pga_source == src) & (allr.pga_level == "municipality")].iloc[0]
            else:
                r = allr[(allr.model == m) & (allr.pga_source == src) & allr.selected].iloc[0]
            h = head[(head.pga_source == src) & (head.model == m)]
            delta = (f"${h.d_rps.iloc[0]:+.4f}$ [${h.d_rps_ci_low.iloc[0]:+.4f}$, ${h.d_rps_ci_high.iloc[0]:+.4f}$], "
                     f"{h.municipalities_better.iloc[0]}" if len(h) else "--")
            name = LABEL[m]                                  # majority uses no PGA: same on both maps (caption)
            cells = [f"{r.rps:.4f}", f"{r.brier_severe:.4f}", f"{r.log_loss:.3f}", f"{r.qwk:.3f}", f"{r.accuracy:.3f}"]
            if r.rps == best_rps[src]:                       # lowest test RPS of the map (not a claim by itself)
                cells[0] = r"\textbf{" + cells[0] + "}"
            lines.append(f"{'Atlas' if src == 'preferred' else 'NEIC'} & {name} & " + " & ".join(cells)
                         + f" & {delta} \\\\")
        if src != list(MAPS)[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines).replace("−", "-") + "\n"
    out = T / "14_table1_final_test.tex"
    out.write_text(tex)
    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    shutil.copy(out, PAPER / "tables" / out.name)
    print(tex)


def main():
    set_style()
    import matplotlib as mpl
    mpl.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300,     # editable text in the PDFs
                         "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
    g = gains()
    g.to_csv(T / "14_fig1_data.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(4).to_string(index=False))
    fig1(g)
    fig2()
    table1()
    (T / "14_run.json").write_text(json.dumps({"figures": ["14_fig1_gain_by_design", "14_fig2_interpretation"], "text_width_in": TEXT_WIDTH_IN}))


if __name__ == "__main__":
    main()
