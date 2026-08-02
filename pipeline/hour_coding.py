"""Compare ways of coding hour-of-day: spline vs one dummy per hour.

Splines fit a smooth curve in 4 parameters. Dummies fit each hour freely in
~15, and — the reason this matters here — a dummy model exports as a lookup
table, which a spline basis does not. The app needs the model to survive as
JSON that TypeScript can add up.

Standard errors are unclustered here; this is a specification comparison, not
an inference table. Production fits cluster by site.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent

BASE = "log_dur + log_dist + bs(doy, df=3) + wind_max + temp_mean + precip_sum"
SPECIES = "y_blujay"


def main():
    m = pd.read_parquet(ROOT / "data" / "pa_model_table.parquet")
    m["hr"] = m.hour.round().astype(int)
    m = m[m.hr.between(5, 20)].copy()
    # drop hours with too few checklists to estimate — 05:00 has n=3, which
    # separates perfectly and sends every other hour's odds ratio to zero
    keep = m.groupby("hr").size()
    m = m[m.hr.isin(keep[keep >= 50].index)].copy()

    def fit(rhs):
        y, X = dmatrices(f"{SPECIES} ~ {rhs}", m, return_type="dataframe")
        return sm.Logit(y, X).fit(disp=0)

    spline = fit(f"{BASE} + bs(hour, df=4)")
    dummy = fit(f"{BASE} + C(hr, Treatment(reference=8))")

    print(f"n = {len(m):,}   species = {SPECIES}\n")
    print(f"{'hour coding':<20}{'params':>8}{'AIC':>10}{'pseudo-R2':>11}")
    print("-" * 49)
    print(f"{'bs(hour, df=4)':<20}{int(spline.df_model):>8}{spline.aic:>10.1f}"
          f"{spline.prsquared:>11.4f}")
    print(f"{'one dummy per hour':<20}{int(dummy.df_model):>8}{dummy.aic:>10.1f}"
          f"{dummy.prsquared:>11.4f}")

    n = m.groupby("hr").size()
    print(f"\n=== hour effect, dummy spec (odds ratio vs 08:00) ===")
    for h in sorted(m.hr.unique()):
        if h == 8:
            print(f"  08:00   OR  1.00    n={n.get(8, 0):>4}   baseline")
            continue
        k = f"C(hr, Treatment(reference=8))[T.{h}]"
        if k not in dummy.params.index:
            continue
        orr, p = np.exp(dummy.params[k]), dummy.pvalues[k]
        thin = " (thin)" if n.get(h, 0) < 50 else ""
        print(f"  {h:02d}:00   OR {orr:5.2f}{'*' if p < 0.05 else ' '}   "
              f"n={n.get(h, 0):>4}   {'#' * max(0, int(orr * 8))}{thin}")


if __name__ == "__main__":
    main()
