"""Wireframe mockups of the two panels, drawn for the brief.

Deliberately flat and unstyled — these communicate structure and content, not
visual design. Colour is used only where it carries meaning (the heatmap, the
best window).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

FIGS = Path(__file__).parent.parent / "figures"
FIGS.mkdir(exist_ok=True)

PETROL, AMBER = "#00897A", "#BE7A12"
INK, MUTED, LINE, FILL = "#1A2224", "#6B7A7C", "#CFD9D8", "#F4F7F6"

mpl.rcParams.update({"figure.dpi": 200, "savefig.bbox": "tight",
                     "font.family": "sans-serif", "font.size": 7})


def box(ax, x, y, w, h, fc="white", ec=LINE, lw=0.8, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw))


def field(ax, x, y, w, label, value, h=0.052):
    """A labelled input field."""
    ax.text(x, y + h + 0.012, label, fontsize=5.6, color=MUTED, va="bottom")
    box(ax, x, y, w, h, fc=FILL)
    ax.text(x + 0.012, y + h / 2, value, fontsize=6.4, color=INK, va="center")


def panel_log():
    fig, ax = plt.subplots(figsize=(3.5, 4.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0, 0, 1, 1, fc="white", ec=LINE, lw=1.1, r=0.02)
    ax.text(0.045, 0.955, "Log a sighting", fontsize=9.5, color=INK, weight="bold", va="center")
    ax.plot([0.045, 0.955], [0.925, 0.925], color=LINE, lw=0.8)

    # map strip
    ax.text(0.045, 0.905, "WHERE", fontsize=5.6, color=MUTED, va="top")
    box(ax, 0.045, 0.775, 0.91, 0.105, fc="#EAF1EF", ec=LINE)
    ax.plot(0.30, 0.828, marker="v", markersize=6, color=AMBER)
    ax.text(0.34, 0.828, "Frick Park  ·  40.4382, −79.9066", fontsize=6.2, color=INK, va="center")
    ax.text(0.93, 0.828, "GPS", fontsize=5.6, color=PETROL, va="center", ha="right", weight="bold")

    field(ax, 0.045, 0.688, 0.44, "DATE", "14 May 2025")
    field(ax, 0.515, 0.688, 0.44, "START TIME", "06:45")

    ax.text(0.045, 0.652, "HOW", fontsize=5.6, color=MUTED, va="top")
    for i, (lab, on) in enumerate([("Stationary", False), ("Traveling", True), ("Incidental", False)]):
        x = 0.045 + i * 0.315
        ax.add_patch(plt.Circle((x + 0.018, 0.607), 0.011, facecolor=PETROL if on else "white",
                                edgecolor=PETROL if on else LINE, linewidth=0.8))
        ax.text(x + 0.042, 0.607, lab, fontsize=6.2, color=INK if on else MUTED, va="center")

    field(ax, 0.045, 0.484, 0.28, "DURATION", "90 min")
    field(ax, 0.355, 0.484, 0.28, "DISTANCE", "1.6 km")
    field(ax, 0.665, 0.484, 0.29, "PARTY", "1")
    ax.text(0.045, 0.462, "required — these are what make the log usable",
            fontsize=5.2, color=AMBER, va="top", style="italic")

    ax.text(0.045, 0.424, "SPECIES", fontsize=5.6, color=MUTED, va="top")
    box(ax, 0.045, 0.275, 0.91, 0.125, fc=FILL)
    for i, (sp, n) in enumerate([("Wood Thrush", "2"), ("Northern Cardinal", "3"),
                                 ("Blue Jay", "1"), ("+ add species", "")]):
        y = 0.373 - i * 0.031
        c = PETROL if sp.startswith("+") else INK
        ax.text(0.065, y, sp, fontsize=6.2, color=c, va="center")
        if n:
            ax.text(0.925, y, n, fontsize=6.2, color=MUTED, va="center", ha="right")

    ax.add_patch(Rectangle((0.045, 0.232), 0.026, 0.026, facecolor=PETROL, edgecolor=PETROL))
    ax.text(0.0575, 0.245, "✓", fontsize=6, color="white", ha="center", va="center")
    ax.text(0.085, 0.252, "Complete list — I reported every species I could identify",
            fontsize=6, color=INK, va="center")
    ax.text(0.085, 0.228, "This is what tells us what wasn't there, not just what was.",
            fontsize=5.2, color=MUTED, va="center", style="italic")

    box(ax, 0.045, 0.155, 0.91, 0.05, fc=PETROL, ec=PETROL)
    ax.text(0.5, 0.18, "Save checklist", fontsize=7, color="white", ha="center", va="center", weight="bold")

    # leaderboard
    ax.plot([0.045, 0.955], [0.125, 0.125], color=LINE, lw=0.8)
    ax.text(0.045, 0.101, "THIS MONTH — DETECTION EFFICIENCY", fontsize=5.6, color=MUTED, va="center")
    for i, (rank, name, val, hrs) in enumerate(
            [("1", "birderA", "127%", "18 h"), ("2", "you", "119%", "4 h"), ("3", "birderC", "94%", "41 h")]):
        y = 0.072 - i * 0.024
        me = name == "you"
        ax.text(0.05, y, rank, fontsize=6, color=MUTED, va="center")
        ax.text(0.10, y, name, fontsize=6.2, color=PETROL if me else INK,
                weight="bold" if me else "normal", va="center")
        ax.text(0.62, y, val, fontsize=6.2, color=INK, va="center", ha="right")
        ax.text(0.94, y, hrs, fontsize=6, color=MUTED, va="center", ha="right")
    ax.text(0.045, -0.002, "ranked by species found vs expected for your effort — not by hours spent",
            fontsize=5.2, color=MUTED, style="italic", va="center")

    fig.savefig(FIGS / "panel1-log.pdf")
    plt.close(fig)


def panel_forecast():
    fig, ax = plt.subplots(figsize=(3.5, 4.7))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    box(ax, 0, 0, 1, 1, fc="white", ec=LINE, lw=1.1, r=0.02)
    ax.text(0.045, 0.955, "Forecast", fontsize=9.5, color=INK, weight="bold", va="center")
    ax.plot([0.045, 0.955], [0.925, 0.925], color=LINE, lw=0.8)

    field(ax, 0.045, 0.828, 0.55, "SPECIES", "Wood Thrush          ▾")
    field(ax, 0.625, 0.828, 0.33, "I HAVE", "1 hour     ▾")
    field(ax, 0.045, 0.742, 0.91, "NEAR", "Pittsburgh, PA                    use my location")

    # hotspots ranked by probability
    ax.text(0.045, 0.708, "NEAREST HOTSPOTS — RANKED BY YOUR ODDS", fontsize=5.6, color=MUTED, va="top")
    for i, (name, km, pct, best) in enumerate(
            [("Frick Park", "3.2 km", "41%", True),
             ("Schenley Park", "4.8 km", "34%", False),
             ("Riverview Park", "6.1 km", "22%", False)]):
        y = 0.638 - i * 0.038
        if best:
            box(ax, 0.045, y - 0.015, 0.91, 0.032, fc="#EAF4F2", ec=PETROL, lw=0.7)
        ax.text(0.065, y, name, fontsize=6.4, color=INK,
                weight="bold" if best else "normal", va="center")
        ax.text(0.55, y, km, fontsize=6, color=MUTED, va="center")
        ax.text(0.93, y, pct, fontsize=6.6, color=PETROL if best else MUTED,
                weight="bold" if best else "normal", va="center", ha="right")

    # heatmap
    ax.text(0.045, 0.512, "FRICK PARK — NEXT 7 DAYS", fontsize=5.6, color=MUTED, va="top")
    days = ["M", "T", "W", "T", "F", "S", "S"]
    hours = ["06", "08", "10", "12", "14", "16", "18", "20"]
    vals = [[38, 40, 39, 41, 36, 40, 39],
            [33, 35, 34, 36, 31, 35, 34],
            [26, 27, 27, 28, 24, 27, 27],
            [17, 18, 18, 19, 16, 18, 18],
            [14, 15, 15, 16, 13, 15, 15],
            [16, 17, 17, 18, 15, 17, 17],
            [21, 22, 22, 23, 19, 22, 22],
            [12, 13, 13, 14, 11, 13, 13]]
    x0, y0, cw, ch = 0.115, 0.16, 0.117, 0.036
    for j, d in enumerate(days):
        ax.text(x0 + j * cw + cw / 2, 0.466, d, fontsize=5.8, color=MUTED, ha="center", va="center")
    for i, h in enumerate(hours):
        yy = 0.446 - i * ch
        ax.text(0.10, yy - ch / 2, h, fontsize=5.8, color=MUTED, ha="right", va="center")
        for j in range(7):
            v = vals[i][j]
            shade = 0.10 + 0.90 * (v - 11) / 30
            best = (i == 0 and j == 3)
            ax.add_patch(Rectangle((x0 + j * cw, yy - ch), cw - 0.006, ch - 0.005,
                                   facecolor=PETROL, alpha=shade,
                                   edgecolor=AMBER if best else "none", linewidth=1.2))
            ax.text(x0 + j * cw + (cw - 0.006) / 2, yy - ch / 2, f"{v}",
                    fontsize=5.4, color="white" if shade > 0.55 else INK,
                    ha="center", va="center", weight="bold" if best else "normal")

    ax.plot([0.045, 0.955], [0.135, 0.135], color=LINE, lw=0.8)
    ax.text(0.045, 0.108, "Best window", fontsize=6, color=MUTED, va="center")
    ax.text(0.045, 0.078, "Thursday 06:00–07:00 — 41%", fontsize=7.6, color=AMBER,
            weight="bold", va="center")
    for i, (lab, pts) in enumerate([("dawn start", "+18"), ("60 min not 15", "+11"),
                                    ("light wind Thu", "+3")]):
        y = 0.044 - i * 0.019
        ax.text(0.045, y, lab, fontsize=5.8, color=MUTED, va="center")
        ax.text(0.40, y, pts, fontsize=5.8, color=PETROL, va="center", ha="right")
    ax.text(0.55, 0.030, "41% of past 1-hour visits here\nat this hour reported it",
            fontsize=5.2, color=MUTED, va="center", style="italic", linespacing=1.6)

    fig.savefig(FIGS / "panel2-forecast.pdf")
    plt.close(fig)


if __name__ == "__main__":
    panel_log()
    panel_forecast()
    print(f"wrote panel1-log.pdf and panel2-forecast.pdf to {FIGS}")
