"""Backfill site coordinates.

The first scrape kept only loc_id from /product/lists and discarded the `loc`
object, which carries latitude and longitude — including for personal
locations, which /ref/hotspot/info will not resolve.

Costs one call per scrape-day (10), not one per location (479), because the
feed response covers every checklist filed that day.

Privacy note: eBird auto-names personal locations after the street address
they sit on. Coordinates are fine to model with; the names must never be
surfaced in a UI.
"""
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
KEY = os.environ["EBIRD_API_KEY"]
REGION = os.getenv("SCRAPE_REGION", "US-PA-003")
DB = ROOT / "data" / "birding.duckdb"

sess = requests.Session()


def fetch_day(date):
    y, m, d = date.split("-")
    r = sess.get(f"https://api.ebird.org/v2/product/lists/{REGION}/{y}/{m}/{d}",
                 headers={"X-eBirdApiToken": KEY},
                 params={"maxResults": 200}, timeout=45)
    r.raise_for_status()
    time.sleep(0.5)
    return r.json()


if __name__ == "__main__":
    # take dates from the database, not the day-list file, so every scraped
    # day is covered even after the day list has been swapped
    con = duckdb.connect(str(DB), read_only=True)
    days = [str(r[0]) for r in
            con.execute("SELECT DISTINCT obs_date FROM checklists ORDER BY 1").fetchall()]
    n_ck = con.execute("SELECT count(*) FROM checklists").fetchone()[0]
    con.close()
    if not days:
        sys.exit(f"no checklists in {DB.name} ({n_ck} rows) — nothing to "
                 f"backfill. Run the scrape first; if it just ran, it stored "
                 f"nothing and that is the real failure.")
    print(f"{n_ck:,} checklists across {len(days)} days", flush=True)
    rows = {}
    for i, date in enumerate(days, 1):
        for item in fetch_day(date):
            loc = item.get("loc") or {}
            lid = loc.get("locId")
            if lid and lid not in rows:
                rows[lid] = (lid, loc.get("latitude"), loc.get("longitude"),
                             bool(loc.get("isHotspot")))
        if i % 5 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {date}  cumulative: {len(rows)}", flush=True)

    df = pd.DataFrame(rows.values(),
                      columns=["loc_id", "lat", "lon", "is_hotspot"])
    con = duckdb.connect(str(DB))
    con.execute("CREATE OR REPLACE TABLE locations AS SELECT * FROM df")
    n_have, n_need = con.execute("""
        SELECT (SELECT count(*) FROM locations),
               (SELECT count(DISTINCT loc_id) FROM checklists)""").fetchone()
    missing = con.execute("""
        SELECT count(DISTINCT c.loc_id) FROM checklists c
        LEFT JOIN locations l USING (loc_id) WHERE l.loc_id IS NULL""").fetchone()[0]
    hot = con.execute("SELECT coalesce(sum(is_hotspot::INT), 0), count(*) "
                      "FROM locations").fetchone()
    con.close()

    print(f"\nresolved {n_have} locations, {n_need} needed, {missing} still missing")
    pct = f"  ({hot[0]/hot[1]:.0%})" if hot[1] else ""
    print(f"hotspots: {hot[0]} of {hot[1]}{pct}")
    print(f"lat range {df.lat.min():.3f} to {df.lat.max():.3f}")
    print(f"lon range {df.lon.min():.3f} to {df.lon.max():.3f}")
    span_km = (df.lat.max()-df.lat.min())*111
    print(f"county span: ~{span_km:.0f} km north-south "
          f"-> {'MORE than' if span_km > 25 else 'within'} one 0.25 deg ERA5 cell")
