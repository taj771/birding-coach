"""Pick scrape dates that maximise weather variation.

Ten API-days is a small budget, so spend it on days that actually differ.
Takes the windiest and calmest day of alternating months across 2025 —
season spread and wind contrast for the same call cost as ten days in a row.

Selecting on the regressor (wind) is fine; selecting on the outcome
(detections) would bias everything. We only ever look at weather here.
"""
import json
from pathlib import Path

import pandas as pd
import requests

# Allegheny County centroid (Pittsburgh)
LAT, LON = 40.44, -79.98
YEAR = 2025
OUT = Path(__file__).parent.parent / "data" / "scrape_days.json"


def daily_weather(year):
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "start_date": f"{year}-01-01", "end_date": f"{year}-10-31",
            "daily": "wind_speed_10m_max,precipitation_sum,temperature_2m_mean",
            "timezone": "America/New_York",
        },
        timeout=60,
    )
    r.raise_for_status()
    d = pd.DataFrame(r.json()["daily"])
    d["time"] = pd.to_datetime(d.time)
    d["month"] = d.time.dt.month
    return d


def pick(d):
    """One windiest + one calmest day per alternating month."""
    days = []
    for m in range(1, 11):
        sub = d[d.month == m].dropna(subset=["wind_speed_10m_max"])
        if sub.empty:
            continue
        # alternate: odd months take the windy extreme, even months the calm one
        row = (sub.loc[sub.wind_speed_10m_max.idxmax()] if m % 2
               else sub.loc[sub.wind_speed_10m_max.idxmin()])
        days.append({
            "date": row.time.strftime("%Y-%m-%d"),
            "wind_max": float(row.wind_speed_10m_max),
            "precip": float(row.precipitation_sum),
            "temp_mean": float(row.temperature_2m_mean),
            "kind": "windy" if m % 2 else "calm",
        })
    return days


if __name__ == "__main__":
    w = daily_weather(YEAR)
    days = pick(w)

    print(f"Allegheny County ({LAT}, {LON}) — {YEAR}\n")
    print(f"{'date':<12} {'kind':<6} {'wind_max':>9} {'precip':>8} {'temp':>7}")
    for x in days:
        print(f"{x['date']:<12} {x['kind']:<6} {x['wind_max']:>9.1f} "
              f"{x['precip']:>8.1f} {x['temp_mean']:>7.1f}")

    winds = [x["wind_max"] for x in days]
    print(f"\nwind range across selected days: {min(winds):.1f} - {max(winds):.1f} km/h")
    print(f"(vs Georgia June sample, where p5-p95 was 2.4 - 17.1)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(days, indent=2))
    print(f"\nwrote {OUT}")
