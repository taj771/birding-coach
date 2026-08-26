"""Is the model we actually publish still honest?

    python pipeline/eval_calibration.py

WHAT THIS IS NOT
calibrate.py fits its own sklearn model on 75% of sites and tests on the rest.
That answers "can a logistic model on these features be calibrated" — a
question about the approach. It says nothing about the coefficients sitting on
the public dataset, which is what the app and the agent actually serve.

This scores the PUBLISHED coefficients. Same flat term lookup as serve/score.ts,
same refusal rules, so the number describes the artifact users get rather than
a model that resembles it.

WHY OUT-OF-SAMPLE IS EASY HERE
The published model carries fitted_at. Every checklist scraped after that
timestamp was unavailable when it was fitted, so it cannot have been trained
on. No splitting, no held-out set to maintain: the passage of time does it, and
the set grows by itself between runs.

Honest limit: this is out-of-sample but NOT out-of-time. The backfill scrapes
historical days, so a checklist scraped yesterday may describe last October.
That still tests generalisation to unseen data; it does not test whether the
world has moved since the fit. Reading a drift in this number as "birding
changed" would be wrong — it means "the model does not generalise to sites and
days it never saw", which is the more basic question anyway.

ONLY WHAT WE WOULD ANSWER
Cells the model refuses in the app are skipped here too: an hour with fewer
than MIN_SUPPORT checklists behind it, and a week with no fitted term. Scoring
them anyway would measure the quality of answers we never give, and would
flatter or damn the product for predictions no user can see.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
DB = ROOT / "data" / "birding.duckdb"
TABLE = ROOT / "data" / "pa_model_table.parquet"
OUT = ROOT / "data" / "calibration.json"

PUBLIC = os.getenv("HF_PUBLIC_DATASET", "taj771/birding-coach-model").strip()
BASE = f"https://huggingface.co/datasets/{PUBLIC}/resolve/main"

# Matches serve/score.ts. An hour the app will not answer for must not be
# scored here either.
MIN_SUPPORT = 50
# Deciles need enough rows per bin to mean anything. Below this a species is
# reported as "too few" rather than given a number that moves on three
# checklists.
MIN_ROWS = 200


def get(path):
    r = requests.get(f"{BASE}/{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def scored(m, c):
    """Predicted probability per row, or NaN where the model would refuse.

    The arithmetic is score.ts line for line. Deliberately not shared with
    fit_logit's own prediction path: the point is to check the published
    numbers, so this reads the published file the same way a browser does.
    """
    hr = m.hr.astype(int)
    wk = m.week.astype(int)

    support = hr.map(lambda h: c.get(f"support_{h:02d}", 0))
    hour_term = hr.map(lambda h: c.get(f"hour_{h:02d}"))
    week_term = wk.map(lambda w: c.get(f"week_{w:02d}"))

    # a site the model never saw falls back to the pooled baseline, exactly as
    # the app does — and is marked, because that number is county-wide
    site_term = m.site.map(lambda s: c.get(f"site_{s}"))
    site_specific = site_term.notna()
    site_term = site_term.fillna(c.get("site_PERSONAL", 0.0))

    logit = (
        c.get("intercept", 0.0)
        + hour_term.fillna(0.0)
        + week_term.fillna(0.0)
        + site_term
        + c.get("log_dur", 0.0) * m.log_dur
        + c.get("log_dist", 0.0) * m.log_dist
        + c.get("n_observers", 0.0) * m.n_observers
        + c.get("traveling", 0.0) * m.traveling
        + c.get("wind_max", 0.0) * m.wind_max
        + c.get("temp_mean", 0.0) * m.temp_mean
        + c.get("precip_sum", 0.0) * m.precip_sum
        + c.get("cloud_mean", 0.0) * m.cloud_mean
    )
    p = 1.0 / (1.0 + np.exp(-logit))

    refused = (support < MIN_SUPPORT) | week_term.isna()
    return p.where(~refused), site_specific, refused


def deciles(p, y):
    """Predicted vs observed in ten bins, the only view that shows a lie.

    A Brier score can look respectable while every prediction is shifted the
    same way. The decile table shows the shape: a model that says 40% and
    delivers 25% is wrong in a way one summary number hides.
    """
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    d = (pd.DataFrame({"pred": p, "obs": y, "q": q})
         .groupby("q").agg(predicted=("pred", "mean"),
                           observed=("obs", "mean"),
                           n=("obs", "size")))
    return d.reset_index(drop=True)


def main():
    if not TABLE.exists():
        sys.exit(f"no {TABLE.name} — run build_model_table.py first.")

    meta = get("meta.json")
    species = get("species.json")
    fitted_at = pd.Timestamp(meta["fitted_at"]).tz_convert("UTC")
    print(f"published model fitted at {fitted_at}")
    print(f"region {meta['region_name']}, {meta['species']} species, "
          f"{len(meta['modelled_weeks'])} of 53 weeks\n")

    m = pd.read_parquet(TABLE)

    # scraped_at is not carried into the model table, so take it from the
    # database. This is the whole hold-out: rows the fit could not have seen.
    con = duckdb.connect(str(DB), read_only=True)
    when = con.execute("select sub_id, scraped_at from checklists").df()
    con.close()
    m = m.merge(when, on="sub_id", how="left")
    m["scraped_at"] = pd.to_datetime(m.scraped_at, utc=True)

    before = len(m)
    m = m[m.scraped_at > fitted_at].copy()
    print(f"{len(m):,} of {before:,} checklists scraped after the fit "
          f"— these are the held-out set")
    if len(m) < MIN_ROWS:
        sys.exit(f"only {len(m)} held-out checklists. Nothing to say yet; "
                 "run again once the backfill has added more.")

    m["hr"] = m.hour.round().astype(int)
    m["week"] = (m.doy // 7).astype(int)
    m["site"] = np.where(m.is_hotspot, m.loc_id, "PERSONAL")

    results = {}
    print(f"{'species':<28}{'n':>8}{'prev':>8}{'Brier':>9}{'gap':>8}")
    print("-" * 61)

    for s in species:
        col = f"y_{s['species_code']}"
        if col not in m.columns:
            continue
        try:
            c = get(f"coef/{s['species_code']}.json")
        except requests.HTTPError:
            continue

        p, site_specific, refused = scored(m, c)
        keep = p.notna()
        if keep.sum() < MIN_ROWS:
            results[s["common_name"]] = {"n": int(keep.sum()), "too_few": True}
            continue

        y = m.loc[keep, col].values
        pp = p[keep].values
        d = deciles(pp, y)
        # mean absolute gap across deciles: the number that answers "when it
        # says 40%, does 40% happen". Brier is reported too, but Brier rewards
        # confident correctness and can improve while calibration worsens.
        gap = float(np.mean(np.abs(d.predicted - d.observed)))
        brier = float(np.mean((pp - y) ** 2))

        results[s["common_name"]] = {
            "species_code": s["species_code"],
            "n": int(keep.sum()),
            "n_refused": int(refused.sum()),
            "share_site_specific": round(float(site_specific[keep].mean()), 3),
            "prevalence": round(float(y.mean()), 4),
            "brier": round(brier, 4),
            "decile_gap": round(gap, 4),
            "deciles": [{"predicted": round(r.predicted, 4),
                         "observed": round(r.observed, 4),
                         "n": int(r.n)} for r in d.itertuples()],
        }
        print(f"{s['common_name']:<28}{keep.sum():>8,}{y.mean():>8.1%}"
              f"{brier:>9.4f}{gap:>8.3f}")

    scored_species = [v for v in results.values() if not v.get("too_few")]
    if not scored_species:
        sys.exit("no species had enough held-out rows to score.")

    overall = float(np.mean([v["decile_gap"] for v in scored_species]))
    payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "model_fitted_at": meta["fitted_at"],
        "held_out_checklists": int(len(m)),
        "species_scored": len(scored_species),
        "mean_decile_gap": round(overall, 4),
        "verdict": ("well calibrated" if overall < 0.03 else
                    "acceptable" if overall < 0.08 else
                    "the numbers are not honest"),
        "note": ("Out-of-sample by scrape time, not out-of-time: the backfill "
                 "collects historical days, so this tests generalisation to "
                 "unseen sites and days rather than drift since the fit."),
        "species": results,
    }
    OUT.write_text(json.dumps(payload, indent=1))

    print(f"\nmean decile gap across {len(scored_species)} species: {overall:.3f}")
    print(f"verdict: {payload['verdict']}")
    print("under 0.03 is well calibrated; over 0.08 means the numbers lie.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
