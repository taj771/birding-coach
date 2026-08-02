"""Are the PA probabilities honest?

The product claims that when the model says 41%, roughly 41% of comparable
outings found the species. That is calibration, and it is the one property the
app cannot do without. Discrimination metrics (AUC) do not test it.

Held out by SITE, not at random: checklists from the same location are not
independent, so a random split would leak and flatter the result.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent

SPECIES = {"blujay": "Blue Jay", "amegfi": "American Goldfinch",
           "rebwoo": "Red-bellied Woodpecker", "sonspa": "Song Sparrow"}
CAT = ["hr", "week"]
NUM = ["log_dur", "log_dist", "n_observers", "traveling",
       "wind_max", "temp_mean", "precip_sum", "cloud_mean", "lat", "lon"]


def prep():
    m = pd.read_parquet(ROOT / "data" / "pa_model_table.parquet")
    m["hr"] = m.hour.round().astype(int)
    keep = m.groupby("hr").size()
    m = m[m.hr.isin(keep[keep >= 50].index)].copy()
    m["week"] = (m.doy // 7).astype(int)
    return m


def fit_predict(m, col, seed=0):
    """Split by site, fit on train, predict on held-out sites."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_i, te_i = next(gss.split(m, groups=m.loc_id))
    tr, te = m.iloc[tr_i], m.iloc[te_i]

    ct = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT),
        ("num", StandardScaler(), NUM)])
    pipe = make_pipeline(ct, LogisticRegression(max_iter=3000, C=1.0))
    pipe.fit(tr[CAT + NUM], tr[col])
    return te[col].values, pipe.predict_proba(te[CAT + NUM])[:, 1]


if __name__ == "__main__":
    m = prep()
    print(f"n = {len(m):,} checklists, {m.loc_id.nunique()} sites")
    print("held out by SITE (25%), so no location appears in both halves\n")

    print(f"{'species':<26}{'prev':>7}{'Brier':>9}{'AUC':>8}{'AUC-PR':>9}")
    print("-" * 59)
    store = {}
    for code, name in SPECIES.items():
        col = f"y_{code}"
        y, p = fit_predict(m, col)
        store[name] = (y, p)
        print(f"{name:<26}{y.mean():>7.1%}{brier_score_loss(y, p):>9.4f}"
              f"{roc_auc_score(y, p):>8.3f}{average_precision_score(y, p):>9.3f}")

    name = "Blue Jay"
    y, p = store[name]
    print(f"\n=== {name} — predicted vs observed, held-out sites ===")
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    cal = (pd.DataFrame({"pred": p, "obs": y, "q": q})
           .groupby("q").agg(pred=("pred", "mean"), obs=("obs", "mean"),
                             n=("obs", "size")))
    print(f"{'decile':>7}{'predicted':>12}{'observed':>11}{'n':>7}   gap")
    for i, r in cal.iterrows():
        gap = r.obs - r.pred
        bar = ("+" if gap > 0 else "-") * min(12, int(abs(gap) * 100))
        print(f"{i+1:>7}{r.pred:>11.1%}{r.obs:>11.1%}{int(r.n):>7}   {bar}")

    mae = float(np.mean(np.abs(cal.pred - cal.obs)))
    print(f"\nmean absolute gap across deciles: {mae:.3f}")
    print("under ~0.03 is well calibrated; over ~0.08 means the numbers lie.")
