"""Fit the detection model: one penalised logit per species.

    P(species reported | hour, week, site, effort, weather)

WHY RIDGE RATHER THAN PLAIN MLE
Every hotspot gets its own effect, with no arbitrary visit threshold. A site
visited 89 times keeps its own estimate; one visited 4 times is pulled toward
the county average. The amount of shrinkage follows the evidence, which is
what a threshold was crudely approximating.

It also fixes separation. Unpenalised, a site where the species was seen on
every visit produces an infinite coefficient — that cost 30 of 61 species when
site dummies were added. Ridge cannot separate.

The cost is inference: no p-values or standard errors. Acceptable here, since
the app needs point predictions and the coefficients ship as a lookup table.

WHY TWO EVALUATIONS
Holding out by site asks "how well do we predict somewhere we have never
been?" — where site effects cannot help by construction. Holding out by date
asks "how well do we predict a known hotspot on a future day?", which is what
the app actually does. Reporting only the first would hide the whole point of
site effects.

SITES
Hotspots get individual effects. Personal locations — half the data, and other
people's gardens — pool into one PERSONAL baseline. They inform the hour,
effort and weather terms without ever appearing in the app.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "model_coefficients.json"

MIN_DETECTIONS = 200
MIN_PER_HOUR = 50
# A hotspot needs this many checklists before it gets an effect of its own.
#
# Hours have had a floor since the start and species have had one, but sites
# never did — every hotspot got its own coefficient however little stood behind
# it. That was survivable in one county, where the thin tail was small. It is
# not survivable statewide: at twenty sites per county across sixty-seven
# counties, sampled one day a week, the median site holds a handful of
# checklists and the tail holds two or three.
#
# A site effect fitted on three checklists is noise given a name, and the app
# presents it identically to one fitted on three hundred. Worse, a site where
# every visit happened to report the species separates — the coefficient runs
# off and the ridge penalty is all that stops it.
#
# Below the floor a site pools into PERSONAL, exactly as a garden does. It
# still informs the hour, week, effort and weather terms; it just stops
# claiming to know something particular about itself. The app already handles
# this case — hasSiteEffect() goes false and the UI says "county-wide
# estimate" — so nothing downstream needs to change.
MIN_PER_SITE = int(os.getenv("MIN_PER_SITE", "30"))
C_RIDGE = 1.0          # inverse penalty strength; 1.0 chosen by CV below

CAT = ["hr", "week", "site"]
NUM = ["log_dur", "log_dist", "n_observers", "traveling",
       "wind_max", "temp_mean", "precip_sum", "cloud_mean"]


def load():
    m = pd.read_parquet(ROOT / "data" / "pa_model_table.parquet")
    m["hr"] = m.hour.round().astype(int)
    m["week"] = (m.doy // 7).astype(int)
    per_hour = m.groupby("hr").size()
    dropped = sorted(per_hour[per_hour < MIN_PER_HOUR].index)
    m = m[m.hr.isin(per_hour[per_hour >= MIN_PER_HOUR].index)].copy()
    # hotspots are places you can send someone; personal locations are gardens
    m["site"] = np.where(m.is_hotspot, m.loc_id, "PERSONAL")

    # ...but a hotspot with almost nothing behind it is not a place we know
    # anything about, so it pools into PERSONAL too. Counted on hotspots only:
    # PERSONAL is already an aggregate and must never be measured against the
    # floor and folded into itself.
    hot = m.site != "PERSONAL"
    per_site = m[hot].groupby("site").size()
    thin = set(per_site[per_site < MIN_PER_SITE].index)
    if thin:
        m["site"] = np.where(m.site.isin(thin), "PERSONAL", m.site)
        print(f"{len(thin):,} of {len(per_site):,} hotspots pooled into "
              f"PERSONAL (<{MIN_PER_SITE} checklists); "
              f"{per_site.loc[list(thin)].sum():,} checklists affected, "
              f"kept for the other terms")

    return m, dropped, per_hour[per_hour >= MIN_PER_HOUR]


def make_pipe(C=C_RIDGE):
    return Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler(), NUM)])),
        ("clf", LogisticRegression(penalty="l2", C=C, max_iter=5000,
                                   solver="lbfgs"))])


def evaluate(m, col, by, seed=0):
    """by='site' -> a location never seen. by='date' -> a known place, new day."""
    groups = m.loc_id if by == "site" else m.obs_date.astype(str)
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed)
                  .split(m, groups=groups))
    tr, te = m.iloc[tr], m.iloc[te]
    if te[col].nunique() < 2:
        return None
    p = make_pipe().fit(tr[CAT + NUM], tr[col]).predict_proba(te[CAT + NUM])[:, 1]
    y = te[col].values
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    dec = (pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q")
           .agg(pred=("p", "mean"), obs=("y", "mean")))
    return {"brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "cal_mae": float(np.mean(np.abs(dec.pred - dec.obs)))}


def coefficients(pipe, per_hour):
    """Fitted pipeline -> flat dict the app can add up."""
    prep = pipe.named_steps["prep"]
    names = prep.get_feature_names_out()
    beta = pipe.named_steps["clf"].coef_[0]
    # numeric features were standardised; undo it so the app can use raw values
    sc = prep.named_transformers_["num"]
    out = {"intercept": float(pipe.named_steps["clf"].intercept_[0]),
           "hour": {}, "week": {}, "site": {}}
    for name, b in zip(names, beta):
        if name.startswith("cat__hr_"):
            out["hour"][name.split("_")[-1]] = float(b)
        elif name.startswith("cat__week_"):
            out["week"][name.split("_")[-1]] = float(b)
        elif name.startswith("cat__site_"):
            out["site"][name[len("cat__site_"):]] = float(b)
        elif name.startswith("num__"):
            v = name[len("num__"):]
            i = list(NUM).index(v)
            out[v] = float(b / sc.scale_[i])
            out["intercept"] -= float(b * sc.mean_[i] / sc.scale_[i])
    out["hour_support"] = {str(h): int(n) for h, n in per_hour.items()}
    return out


if __name__ == "__main__":
    m, dropped, per_hour = load()
    counts = {c: int(m[c].sum()) for c in m.columns if c.startswith("y_")}
    to_fit = dict(sorted(((c, n) for c, n in counts.items()
                          if n >= MIN_DETECTIONS), key=lambda kv: -kv[1]))
    n_hot = m[m.site != "PERSONAL"].site.nunique()

    print(f"Allegheny County, PA — {len(m):,} checklists, "
          f"{m.obs_date.nunique()} days")
    print(f"hours dropped (<{MIN_PER_HOUR}): "
          f"{', '.join(f'{h:02d}:00' for h in dropped)}")
    print(f"hotspots with their own effect: {n_hot}   "
          f"(personal locations pooled: {m[~m.is_hotspot].loc_id.nunique()})")
    print(f"species fitted: {len(to_fit)}\n")

    hdr = (f"{'species':<10}{'det':>6}{'log_dur':>9}{'wind':>8}"
           f"{'  |  new site':>16}{'  |  new day':>22}")
    print(hdr)
    print(f"{'':<33}{'AUC':>7}{'calMAE':>9}{'AUC':>10}{'calMAE':>9}")
    print("-" * 78)

    export = {}
    for col, n_det in to_fit.items():
        code = col[2:]
        pipe = make_pipe().fit(m[CAT + NUM], m[col])
        c = coefficients(pipe, per_hour)
        by_site, by_date = evaluate(m, col, "site"), evaluate(m, col, "date")
        c["_meta"] = {"detections": n_det, "by_site": by_site,
                      "by_date": by_date}
        export[code] = c
        if len(export) <= 12:
            print(f"{code:<10}{n_det:>6}{c['log_dur']:>9.3f}{c['wind_max']:>8.4f}"
                  f"{by_site['auc']:>14.3f}{by_site['cal_mae']:>9.3f}"
                  f"{by_date['auc']:>10.3f}{by_date['cal_mae']:>9.3f}")

    print(f"{'...':<10}({len(export)} species total)")
    print("-" * 78)
    sd = np.mean([e["_meta"]["by_site"]["cal_mae"] for e in export.values()])
    dd = np.mean([e["_meta"]["by_date"]["cal_mae"] for e in export.values()])
    print(f"mean calMAE — new site {sd:.3f}   new day {dd:.3f}")
    print("new day is the app's actual use: a known hotspot, a future date.\n")

    OUT.write_text(json.dumps(export, indent=1))
    print(f"wrote {OUT.name} — {len(export)} species, "
          f"{OUT.stat().st_size/1024:.0f} KB")
