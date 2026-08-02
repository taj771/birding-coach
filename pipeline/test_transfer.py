"""Cross-region test: does the hour-of-day curve transfer?

The brief claims dawn chorus and search effort are *behaviour*, so the
conditions layer transfers between regions while occurrence does not. That is
the load-bearing claim of the whole architecture, and it is cheap to check.

Georgia    Wood Thrush detection rate by hour   (47,254 checklists, June)
Allegheny  species richness by hour             (994 checklists, 10 days)

Different outcomes on different scales, so compare the *shape* after
controlling for effort — index each curve to its own peak.
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
pd.set_option("display.width", 200)


def allegheny():
    con = duckdb.connect(str(ROOT / "data" / "birding.duckdb"), read_only=True)
    d = con.execute("""
        SELECT sub_id, loc_id, obs_date, obs_time, duration_hrs, distance_km,
               n_observers, protocol_id, richness
        FROM checklists
        WHERE all_reported AND richness > 0
          AND duration_hrs BETWEEN 0.08 AND 5
          AND n_observers <= 10
          AND obs_time IS NOT NULL
    """).df()
    con.close()
    t = pd.to_datetime(d.obs_time, format="%H:%M", errors="coerce")
    d["hour"] = t.dt.hour + t.dt.minute / 60
    d["hr"] = d.hour.round().astype("Int64")
    d["date"] = pd.to_datetime(d.obs_date)
    d["doy"] = d.date.dt.dayofyear
    d["log_dur"] = np.log(d.duration_hrs)
    d["log_dist"] = np.log1p(d.distance_km.fillna(0))
    return d.dropna(subset=["hr"])


def georgia():
    g = pd.read_csv(ROOT / "data" / "checklists-zf_woothr_jun_us-ga.csv")
    g["y"] = g.species_observed.astype(int)
    g["hr"] = g.hours_of_day.round().astype(int)
    return g


def curve(series_by_hour, lo=5, hi=20):
    """Index a curve to its own peak so shapes are comparable across scales."""
    s = series_by_hour.reindex(range(lo, hi + 1)).dropna()
    return s / s.max()


if __name__ == "__main__":
    a = allegheny()
    g = georgia()
    print(f"Allegheny  {len(a):>6,} checklists, {a.loc_id.nunique()} sites, "
          f"mean richness {a.richness.mean():.1f}")
    print(f"Georgia    {len(g):>6,} checklists, Wood Thrush prevalence {g.y.mean():.1%}\n")

    # ---- 1. raw shapes -------------------------------------------------
    a_raw = a[a.hr.between(5, 20)].groupby("hr").richness.agg(["mean", "size"])
    g_raw = g[g.hr.between(5, 20)].groupby("hr").y.agg(["mean", "size"])

    print("=== hour-of-day, raw (indexed to each region's own peak) ===")
    print(f"{'hr':>3}  {'ALLEGHENY richness':>20} {'n':>5}   {'GEORGIA detection':>19} {'n':>6}")
    ca, cg = curve(a_raw["mean"]), curve(g_raw["mean"])
    for h in range(5, 21):
        av = f"{a_raw['mean'].get(h, np.nan):5.1f} sp {ca.get(h, np.nan):5.2f}" if h in a_raw.index else " " * 14
        an = f"{int(a_raw['size'].get(h, 0)):>5}"
        gv = f"{g_raw['mean'].get(h, np.nan):6.1%} {cg.get(h, np.nan):5.2f}" if h in g_raw.index else " " * 13
        gn = f"{int(g_raw['size'].get(h, 0)):>6}"
        print(f"{h:>3}  {av:>20} {an}   {gv:>19} {gn}")

    both = pd.concat([ca.rename("alleg"), cg.rename("ga")], axis=1).dropna()
    print(f"\ncorrelation of indexed curves: r = {both.alleg.corr(both.ga):.3f}  (n={len(both)} hours)")
    print(f"Allegheny peak hour {ca.idxmax():02d}:00   Georgia peak hour {cg.idxmax():02d}:00")

    # ---- 2. effort-adjusted hour curve (Allegheny) ---------------------
    print("\n=== Allegheny, effort-adjusted (negative binomial) ===")
    a2 = a[a.hr.between(5, 20)].copy()
    a2["hrf"] = a2.hr.astype(int).astype(str)
    m = smf.glm("richness ~ C(hrf, Treatment(reference='10')) + log_dur + log_dist "
                "+ n_observers + doy",
                data=a2, family=sm.families.NegativeBinomial()).fit(
                    cov_type="cluster", cov_kwds={"groups": a2.loc_id})
    print(f"  log_dur beta = {m.params['log_dur']:+.3f}  "
          f"(p={m.pvalues['log_dur']:.2g})  -> doubling time gives "
          f"{2**m.params['log_dur']:.2f}x species")
    print("  hour effects vs 10:00 (exp beta):")
    for k in sorted([k for k in m.params.index if k.startswith("C(hrf")],
                    key=lambda x: int(x.split("T.")[1].rstrip("]"))):
        h = int(k.split("T.")[1].rstrip("]"))
        star = "*" if m.pvalues[k] < 0.05 else " "
        bar = "#" * max(0, int((np.exp(m.params[k]) - 0.6) * 30))
        print(f"    {h:02d}:00  {np.exp(m.params[k]):5.2f}{star}  {bar}")

    # ---- 3. the designed wind contrast ---------------------------------
    print("\n=== wind contrast (the reason those 10 days were chosen) ===")
    days = pd.read_json(ROOT / "data" / "scrape_days.json")
    days["date"] = pd.to_datetime(days.date)
    a3 = a.merge(days[["date", "wind_max", "kind", "temp_mean"]], on="date", how="inner")
    print(a3.groupby("kind").agg(checklists=("sub_id", "size"),
                                 mean_richness=("richness", "mean"),
                                 mean_wind=("wind_max", "mean")).round(2).to_string())
    mw = smf.glm("richness ~ wind_max + log_dur + log_dist + n_observers + doy + hour",
                 data=a3, family=sm.families.NegativeBinomial()).fit(
                     cov_type="cluster", cov_kwds={"groups": a3.loc_id})
    b, p = mw.params["wind_max"], mw.pvalues["wind_max"]
    lo, hi = a3.wind_max.min(), a3.wind_max.max()
    print(f"\n  wind_max beta = {b:+.4f}  (p={p:.3f})")
    print(f"  across observed range {lo:.1f} -> {hi:.1f} km/h: "
          f"{np.exp(b*(hi-lo)):.2f}x species")
