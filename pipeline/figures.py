"""Build the appendix figures from the data, not from transcribed numbers.

Everything here recomputes from data/checklists-zf_woothr_jun_us-ga.csv so the
figures can never drift from the text.

Palette validated with the dataviz six-checks (light surface, categorical):
    #00897A petrol  ·  #BE7A12 amber   -> all checks pass
"""
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data" / "checklists-zf_woothr_jun_us-ga.csv"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

PETROL, AMBER = "#00897A", "#BE7A12"
INK, MUTED, RULE = "#1A2224", "#5C6A6D", "#D8E0DF"

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 8.5,
    "axes.edgecolor": RULE,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 9.5,
    "axes.titleweight": "regular",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load():
    d = pd.read_csv(DATA)
    d["y"] = d.species_observed.astype(int)
    return d


def bare(ax, xgrid=True):
    """Recessive grid, no chartjunk."""
    ax.grid(axis="x" if xgrid else "y", color=RULE, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)


def fig_hour(d):
    """Detection rate by start hour. Peak emphasised; thin-n bars muted."""
    d = d.copy()
    d["hr"] = d.hours_of_day.round().astype(int)
    g = d[d.hr.between(4, 20)].groupby("hr").y.agg(["mean", "size"])
    peak = g["mean"].idxmax()

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    for hr, r in g.iterrows():
        thin = r["size"] < 100
        ax.barh(hr, r["mean"] * 100,
                color=AMBER if hr == peak else PETROL,
                alpha=0.35 if thin else 1.0, height=0.72)
    # direct-label only the points that carry the argument
    for hr in (peak, g["mean"].idxmin(), 20):
        v = g.loc[hr, "mean"] * 100
        ax.text(v + 0.45, hr, f"{v:.1f}%", va="center", fontsize=7.5,
                color=AMBER if hr == peak else MUTED,
                weight="bold" if hr == peak else "normal")

    ax.set_yticks(range(4, 21))
    ax.set_yticklabels([f"{h:02d}" for h in range(4, 21)])
    ax.set_ylabel("checklist start hour (local)")
    ax.invert_yaxis()
    ax.set_xlabel("checklists reporting the species (%)")
    ax.set_xlim(0, 27)
    ax.set_title("Detection rate by start hour", loc="left", pad=8)
    bare(ax)
    ax.annotate("faded bars: n < 100", xy=(0.98, 0.06), xycoords="axes fraction",
                ha="right", fontsize=6.8, color=MUTED, style="italic")
    fig.savefig(FIGS / "fig1-hour.pdf")
    plt.close(fig)
    return g


def fig_effort(d):
    """Detection rate by effort band. Shows the saturating accumulation curve."""
    d = d.copy()
    bins = [0, .5, 1, 2, 3, 5, 100]
    labs = ["<30m", "30–60m", "1–2h", "2–3h", "3–5h", "5h+"]
    d["band"] = pd.cut(d.effort_hours, bins, labels=labs)
    g = d.groupby("band", observed=True).y.agg(["mean", "size"])

    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    for i, (b, r) in enumerate(g.iterrows()):
        ax.bar(i, r["mean"] * 100, color=PETROL, width=0.68,
               alpha=0.35 if r["size"] < 200 else 1.0)
        ax.text(i, r["mean"] * 100 + 0.5, f"{r['mean']*100:.1f}",
                ha="center", fontsize=7.5, color=MUTED)

    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index)
    ax.set_ylabel("detection rate (%)")
    ax.set_ylim(0, 25)
    ax.set_title("Detection rate by time spent birding", loc="left", pad=8)
    bare(ax, xgrid=False)
    fig.savefig(FIGS / "fig2-effort.pdf")
    plt.close(fig)
    return g


def fig_calibration(d):
    """Predicted vs observed on the held-out split, against the identity line."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    d = d.copy()
    d["log_dur"] = np.log1p(d.effort_hours)
    d["log_dist"] = np.log1p(d.effort_distance_km)
    d["trav"] = (d.protocol_type == "Traveling").astype(int)
    d = d.join(d.groupby("observer_id").size().rename("obs_n"), on="observer_id")
    d["log_obsn"] = np.log(d.obs_n)

    SPL = ["hours_of_day", "day_of_year", "latitude", "longitude"]
    LIN = ["log_dur", "log_dist", "number_observers", "trav", "log_obsn"]
    tr, te = d[d.type == "train"], d[d.type == "test"]

    ct = ColumnTransformer([("sp", SplineTransformer(n_knots=6, degree=3), SPL),
                            ("num", StandardScaler(), LIN)])
    pipe = make_pipeline(ct, LogisticRegression(max_iter=2000)).fit(tr[SPL + LIN], tr.y)
    p = pipe.predict_proba(te[SPL + LIN])[:, 1]

    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"pred": p, "obs": te.y.values, "q": q})
           .groupby("q").agg(pred=("pred", "mean"), obs=("obs", "mean")))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    lim = max(cal.pred.max(), cal.obs.max()) * 100 * 1.12
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=(0, (3, 3)), alpha=0.7,
            zorder=1, label="perfect calibration")
    ax.plot(cal.pred * 100, cal.obs * 100, color=PETROL, lw=1.4, zorder=2)
    ax.scatter(cal.pred * 100, cal.obs * 100, s=26, color=PETROL,
               edgecolor="white", linewidth=1.1, zorder=3)

    ax.set_xlabel("predicted (%)")
    ax.set_ylabel("observed (%)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("Calibration, held-out split", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.6)
    bare(ax)
    bare(ax, xgrid=False)

    brier = brier_score_loss(te.y, p)
    aucpr = average_precision_score(te.y, p)
    ax.annotate(f"Brier {brier:.4f}\nAUC-PR {aucpr:.3f}  (prevalence {te.y.mean():.3f})",
                xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=6.8, color=MUTED, linespacing=1.5)
    fig.savefig(FIGS / "fig3-calibration.pdf")
    plt.close(fig)
    return cal, brier, aucpr, len(te), te.y.mean()


if __name__ == "__main__":
    d = load()
    print(f"loaded {len(d):,} checklists, {d.y.sum():,} detections "
          f"({d.y.mean():.1%} prevalence)")

    g1 = fig_hour(d)
    print(f"  fig1  peak {g1['mean'].idxmax():02d}:00 {g1['mean'].max():.1%}  "
          f"trough {g1['mean'].idxmin():02d}:00 {g1['mean'].min():.1%}  "
          f"ratio {g1['mean'].max()/g1['mean'].min():.1f}x")

    g2 = fig_effort(d)
    print(f"  fig2  {g2['mean'].iloc[0]:.1%} -> {g2['mean'].max():.1%}  "
          f"ratio {g2['mean'].max()/g2['mean'].iloc[0]:.1f}x")

    cal, brier, aucpr, n_te, prev = fig_calibration(d)
    print(f"  fig3  n={n_te:,}  Brier {brier:.4f}  AUC-PR {aucpr:.3f} "
          f"vs prevalence {prev:.3f}")
    print(f"\nwrote 3 PDFs to {FIGS}")
