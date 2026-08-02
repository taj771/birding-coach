"""Build the model one variable at a time and watch the answer move.

Each step adds one block and refits. Tracking the wind and rain coefficients
down the sequence shows which controls actually matter and which rival
explanation each one shuts down.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent


def load():
    d = pd.read_csv(ROOT / "data" / "checklists-zf_woothr_jun_us-ga.csv")
    d["clat"] = (d.latitude / 0.25).round() * 0.25
    d["clon"] = (d.longitude / 0.25).round() * 0.25
    d["date"] = pd.to_datetime(d.observation_date).dt.date
    d["hr"] = d.hours_of_day.round().clip(0, 23).astype(int)

    w = pd.read_parquet(ROOT / "data" / "ga_weather.parquet")
    w["t"] = pd.to_datetime(w.time)
    w["date"], w["hr"] = w.t.dt.date, w.t.dt.hour
    m = d.merge(w.drop(columns=["time", "t"]), on=["clat", "clon", "date", "hr"])

    m["y"] = m.species_observed.astype(int)
    m["wind"] = m.wind_speed_10m
    m["rain"] = (m.precipitation > 0).astype(int)
    m["log_dur"] = np.log1p(m.effort_hours)
    m["log_dist"] = np.log1p(m.effort_distance_km)
    m["trav"] = (m.protocol_type == "Traveling").astype(int)
    m = m.join(m.groupby("observer_id").size().rename("obs_n"), on="observer_id")
    m["log_obsn"] = np.log(m.obs_n)
    return m


STEPS = [
    ("1. weather alone",        "y ~ wind + rain"),
    ("2. + how long they bird", "y ~ wind + rain + log_dur"),
    ("3. + how far, how many",  "y ~ wind + rain + log_dur + log_dist + number_observers + trav"),
    ("4. + TIME OF DAY",        "y ~ wind + rain + log_dur + log_dist + number_observers + trav"
                                " + bs(hours_of_day, df=6)"),
    ("5. + season",             "y ~ wind + rain + log_dur + log_dist + number_observers + trav"
                                " + bs(hours_of_day, df=6) + bs(day_of_year, df=4)"),
    ("6. + where they were",    "y ~ wind + rain + log_dur + log_dist + number_observers + trav"
                                " + bs(hours_of_day, df=6) + bs(day_of_year, df=4)"
                                " + bs(latitude, df=4) + bs(longitude, df=4)"),
    ("7. + observer skill",     "y ~ wind + rain + log_dur + log_dist + number_observers + trav"
                                " + bs(hours_of_day, df=6) + bs(day_of_year, df=4)"
                                " + bs(latitude, df=4) + bs(longitude, df=4) + log_obsn"),
]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "   "


if __name__ == "__main__":
    m = load()
    print(f"Wood Thrush, Georgia, June. n = {len(m):,} checklists, "
          f"{m.y.sum():,} detections ({m.y.mean():.1%})\n")
    print(f"{'model':<26} {'WIND beta':>12} {'':>4} {'RAIN beta':>12} {'':>4} {'AIC':>10}")
    print("-" * 74)

    for label, formula in STEPS:
        y, X = dmatrices(formula, m, return_type="dataframe")
        r = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                               cov_kwds={"groups": m.locality_id})
        bw, pw = r.params["wind"], r.pvalues["wind"]
        br, pr = r.params["rain"], r.pvalues["rain"]
        print(f"{label:<26} {bw:>+12.4f} {stars(pw)} {br:>+12.4f} {stars(pr)} {r.aic:>10.0f}")

    print("-" * 74)
    print("*** p<0.001   ** p<0.01   * p<0.05")
