"""One prediction, shown term by term.

Fits the dummy-coded hour model, then walks through the arithmetic for a
single outing so the numbers in the app can be traced back to something.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrices

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent

SPECIES = "y_blujay"
F = (f"{SPECIES} ~ C(hr, Treatment(reference=8)) + log_dur + log_dist "
     "+ bs(doy, df=3) + wind_max + temp_mean + precip_sum")


def main():
    m = pd.read_parquet(ROOT / "data" / "pa_model_table.parquet")
    m["hr"] = m.hour.round().astype(int)
    keep = m.groupby("hr").size()
    m = m[m.hr.isin(keep[keep >= 50].index)].copy()

    y, X = dmatrices(F, m, return_type="dataframe")
    r = sm.Logit(y, X).fit(disp=0)

    # a fixed outing: mid-May, 1 hour, 1 km, 12 km/h wind, 18 C, no rain
    base = dict(log_dur=np.log(1.0), log_dist=np.log1p(1.0), doy=134,
                wind_max=12.0, temp_mean=18.0, precip_sum=0.0)

    print(f"species: {SPECIES}   n = {len(m):,}\n")
    print("A one-hour walk, 1 km, mid-May, wind 12 km/h, 18 C, dry.\n")
    print(f"{'hour':>6}{'hour term':>12}{'total':>10}{'probability':>14}")
    print("-" * 42)

    for h in [6, 7, 8, 10, 12, 14, 19]:
        row = pd.DataFrame([{**base, "hr": h}])
        # reuse the FITTED design, so categories and spline knots match
        Xn = build_design_matrices([X.design_info], row, return_type="dataframe")[0]
        total = float(np.dot(Xn.values, r.params.values)[0])
        key = f"C(hr, Treatment(reference=8))[T.{h}]"
        term = 0.0 if h == 8 else r.params.get(key, np.nan)
        p = 1 / (1 + np.exp(-total))
        mark = "   <- baseline hour" if h == 8 else ""
        print(f"{h:>4}:00{term:>12.3f}{total:>10.3f}{p:>13.1%}{mark}")

    print(f"\nintercept (everything at baseline): {r.params['Intercept']:.3f}")
    print("\nthe 8am row has hour term 0.000 — its level comes from the")
    print("intercept plus the other terms, not from an hour adjustment.")


if __name__ == "__main__":
    main()
