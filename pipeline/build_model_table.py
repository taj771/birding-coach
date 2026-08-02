"""Build the PA modelling table: one row per checklist, y + all predictors.

Weather is joined per checklist: each site's coordinates are snapped to the
nearest 0.25° ERA5 cell and that cell's hourly series is used. An earlier
version used one county-centroid series for everything; the county actually
spans ~53 km, i.e. several cells, so that erased real spatial variation and
attenuated the weather coefficients.

Weather is aggregated over the hours a checklist actually spans, not just its
start hour — max wind (a gust masks song; the mean hides it), summed precip,
mean temp.
"""
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
WX = ROOT / "data" / "pa_weather_cells.parquet"
OUT = ROOT / "data" / "pa_model_table.parquet"

VARS = "temperature_2m,precipitation,wind_speed_10m,cloud_cover,relative_humidity_2m,pressure_msl"


def fetch_weather(cells, year=2025):
    """One hourly series per 0.25 deg grid cell."""
    if WX.exists():
        return pd.read_parquet(WX)
    frames = []
    for clat, clon in cells:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={"latitude": clat, "longitude": clon,
                    "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
                    "hourly": VARS, "timezone": "America/New_York"},
            timeout=90)
        r.raise_for_status()
        f = pd.DataFrame(r.json()["hourly"])
        f["clat"], f["clon"] = clat, clon
        frames.append(f)
        print(f"    cell {clat:.2f},{clon:.2f}  {len(f):,} hours", flush=True)
    w = pd.concat(frames, ignore_index=True)
    w["ts"] = pd.to_datetime(w.time)
    w = w.drop(columns=["time"])
    w.to_parquet(WX)
    return w


def checklists():
    con = duckdb.connect(str(ROOT / "data" / "birding.duckdb"), read_only=True)
    c = con.execute("""
        SELECT c.sub_id, c.loc_id, c.obs_date, c.obs_time, c.duration_hrs,
               c.distance_km, c.n_observers, c.protocol_id,
               l.lat, l.lon, l.is_hotspot
        FROM checklists c JOIN locations l USING (loc_id)
        WHERE c.all_reported
          AND c.protocol_id IN ('P21','P22')
          AND c.duration_hrs BETWEEN 0.08 AND 5
          AND c.obs_time IS NOT NULL
          AND c.n_observers <= 10
    """).df()
    # richness excluding spuh/slash codes (y00xxx are not species)
    rich = con.execute("""
        SELECT sub_id, count(*) AS richness
        FROM observations WHERE species_code NOT LIKE 'y00%'
        GROUP BY sub_id
    """).df()
    obs = con.execute("""
        SELECT DISTINCT sub_id, species_code FROM observations
        WHERE species_code NOT LIKE 'y00%'
    """).df()
    con.close()
    c = c.merge(rich, on="sub_id", how="left").fillna({"richness": 0})
    return c, obs


def window_weather(c, w):
    """Aggregate weather over the hours each checklist spans, in ITS OWN cell."""
    t = pd.to_datetime(c.obs_time, format="%H:%M", errors="coerce")
    c["hour"] = t.dt.hour + t.dt.minute / 60
    c["clat"] = (c.lat / 0.25).round() * 0.25
    c["clon"] = (c.lon / 0.25).round() * 0.25
    start = pd.to_datetime(c.obs_date) + pd.to_timedelta(c.hour, unit="h")
    end = start + pd.to_timedelta(c.duration_hrs, unit="h")
    by_cell = {k: g.set_index("ts").sort_index() for k, g in w.groupby(["clat", "clon"])}

    rows = []
    for s, e, cl, cn in zip(start, end, c.clat, c.clon):
        seg = by_cell[(cl, cn)].loc[s.floor("h"): e.ceil("h")]
        rows.append({
            "wind_max": seg.wind_speed_10m.max(),
            "precip_sum": seg.precipitation.sum(),
            "temp_mean": seg.temperature_2m.mean(),
            "cloud_mean": seg.cloud_cover.mean(),
            "humid_mean": seg.relative_humidity_2m.mean(),
            "n_wx_hours": len(seg),
        })
    return pd.concat([c.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    c, obs = checklists()
    print(f"checklists after filters: {len(c):,}")

    cells = sorted(set(zip((c.lat / 0.25).round() * 0.25,
                           (c.lon / 0.25).round() * 0.25)))
    print(f"distinct 0.25deg cells: {len(cells)}")
    w = fetch_weather(cells)
    print(f"weather rows: {len(w):,}")

    m = window_weather(c, w)
    m["date"] = pd.to_datetime(m.obs_date)
    m["doy"] = m.date.dt.dayofyear
    m["log_dur"] = np.log(m.duration_hrs)
    m["log_dist"] = np.log1p(m.distance_km.fillna(0))
    m["traveling"] = (m.protocol_id == "P22").astype(int)

    # per-species 0/1 columns for every species with enough detections to
    # support its own model. 200 is the floor used by fit_logit.py; below it
    # a species is fitted on noise.
    MIN_DETECTIONS = 200
    counts = obs.groupby("species_code").size().sort_values(ascending=False)
    top = counts[counts >= MIN_DETECTIONS].index.tolist()
    print(f"species with >={MIN_DETECTIONS} detections: {len(top)} "
          f"(of {len(counts)} recorded)")
    seen = {sp: set(obs[obs.species_code == sp].sub_id) for sp in top}
    for sp in top:
        m[f"y_{sp}"] = m.sub_id.isin(seen[sp]).astype(int)

    m.to_parquet(OUT)
    print(f"\nwrote {OUT}  ({len(m):,} rows, {m.shape[1]} cols)")
    print(f"\nweather spread actually available:")
    for v in ["wind_max", "temp_mean", "precip_sum", "cloud_mean"]:
        print(f"  {v:12} min {m[v].min():7.1f}   med {m[v].median():7.1f}   max {m[v].max():7.1f}")
    print(f"\nper-species prevalence (top 8):")
    for sp in top[:8]:
        print(f"  {sp:8} {m[f'y_{sp}'].mean():6.1%}   ({m[f'y_{sp}'].sum():>3} of {len(m)})")
