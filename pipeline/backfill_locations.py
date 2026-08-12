"""Backfill site coordinates.

The first scrape kept only loc_id from /product/lists and discarded the `loc`
object, which carries latitude and longitude — including for personal
locations, which /ref/hotspot/info will not resolve.

One call per day needing work, not one per location, because the feed response
covers every checklist filed that day.

INCREMENTAL, AND IT HAS TO BE
This used to rebuild the table from scratch, which meant re-fetching every day
in the database on every invocation. backfill.py calls it after each 15-day
chunk, so a 200-day backfill spent roughly 1,575 feed calls re-resolving
locations it already had — and eBird answered that with 429s until the job
died. Only days holding an unresolved loc_id are fetched now, so a second run
over settled data costs nothing at all.

Privacy note: eBird auto-names personal locations after the street address
they sit on. Coordinates are fine to model with; the names must never be
surfaced in a UI.
"""
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))
import ebird_api                                          # noqa: E402

REGION = os.getenv("SCRAPE_REGION", "US-PA-003")
DB = ROOT / "data" / "birding.duckdb"

LOCATIONS = """
CREATE TABLE IF NOT EXISTS locations (
    loc_id      VARCHAR,
    lat         DOUBLE,
    lon         DOUBLE,
    is_hotspot  BOOLEAN
);
"""


if __name__ == "__main__":
    con = duckdb.connect(str(DB))
    con.execute(LOCATIONS)

    n_ck = con.execute("SELECT count(*) FROM checklists").fetchone()[0]
    if not n_ck:
        con.close()
        sys.exit(f"no checklists in {DB.name} — nothing to backfill. Run the "
                 f"scrape first; if it just ran, it stored nothing and that is "
                 f"the real failure.")

    # Only days that still contain a loc_id we cannot place. A day whose sites
    # are all known tells us nothing we do not already have, and asking anyway
    # is what got the job rate limited.
    days = [str(r[0]) for r in con.execute("""
        SELECT DISTINCT c.obs_date
        FROM checklists c
        WHERE NOT EXISTS (SELECT 1 FROM locations l WHERE l.loc_id = c.loc_id)
        ORDER BY 1""").fetchall()]

    n_have, n_need = con.execute("""
        SELECT (SELECT count(*) FROM locations),
               (SELECT count(DISTINCT loc_id) FROM checklists)""").fetchone()
    print(f"{n_ck:,} checklists, {n_need:,} distinct sites, "
          f"{n_have:,} already resolved", flush=True)

    if not days:
        print("every site already has coordinates — no API calls needed.")
    else:
        print(f"{len(days)} day(s) hold unresolved sites\n", flush=True)

    rows = {}
    failed = []
    for i, date in enumerate(days, 1):
        feed = ebird_api.day_feed(REGION, date)
        if feed is None:
            # A day we could not fetch stays unresolved and is retried next
            # run, because the query above is driven by the database rather
            # than by a list of days we decided on up front.
            failed.append(date)
            print(f"  [{i}/{len(days)}] {date}  FEED FAILED", flush=True)
            continue
        for item in feed:
            loc = item.get("loc") or {}
            lid = loc.get("locId")
            if lid and lid not in rows:
                rows[lid] = (lid, loc.get("latitude"), loc.get("longitude"),
                             bool(loc.get("isHotspot")))
        if i % 5 == 0 or i == len(days):
            print(f"  [{i}/{len(days)}] {date}  cumulative: {len(rows)}", flush=True)

    if rows:
        df = pd.DataFrame(rows.values(),
                          columns=["loc_id", "lat", "lon", "is_hotspot"])
        # Anti-join insert rather than INSERT OR IGNORE: the table predates
        # this script and has no primary key on older databases, so OR IGNORE
        # would quietly duplicate instead of skipping. Columns named, never
        # positional — the same lesson the checklists insert already carries.
        con.execute("""
            INSERT INTO locations (loc_id, lat, lon, is_hotspot)
            SELECT d.loc_id, d.lat, d.lon, d.is_hotspot
            FROM df d
            WHERE NOT EXISTS (SELECT 1 FROM locations l WHERE l.loc_id = d.loc_id)
        """)
        print(f"\ninserted {con.execute('SELECT changes()').fetchone()[0]} new sites")
    else:
        df = pd.DataFrame(columns=["loc_id", "lat", "lon", "is_hotspot"])

    if failed:
        print(f"{len(failed)} day(s) could not be fetched and will be retried "
              f"next run: {', '.join(failed[:5])}"
              f"{' ...' if len(failed) > 5 else ''}")
    n_have, n_need = con.execute("""
        SELECT (SELECT count(*) FROM locations),
               (SELECT count(DISTINCT loc_id) FROM checklists)""").fetchone()
    missing = con.execute("""
        SELECT count(DISTINCT c.loc_id) FROM checklists c
        LEFT JOIN locations l USING (loc_id) WHERE l.loc_id IS NULL""").fetchone()[0]
    hot = con.execute("SELECT coalesce(sum(is_hotspot::INT), 0), count(*) "
                      "FROM locations").fetchone()
    # Read the extent from the table, not from df — df now holds only the rows
    # this run added, which on a settled database is none of them.
    extent = con.execute("SELECT min(lat), max(lat), min(lon), max(lon) "
                         "FROM locations WHERE lat IS NOT NULL").fetchone()
    con.close()

    print(f"\nresolved {n_have} locations, {n_need} needed, {missing} still missing")
    pct = f"  ({hot[0]/hot[1]:.0%})" if hot[1] else ""
    print(f"hotspots: {hot[0]} of {hot[1]}{pct}")

    if extent and extent[0] is not None:
        lat_min, lat_max, lon_min, lon_max = extent
        print(f"lat range {lat_min:.3f} to {lat_max:.3f}")
        print(f"lon range {lon_min:.3f} to {lon_max:.3f}")
        span_km = (lat_max - lat_min) * 111
        print(f"county span: ~{span_km:.0f} km north-south "
              f"-> {'MORE than' if span_km > 25 else 'within'} one 0.25 deg ERA5 cell")

    # A site with no coordinates drops its checklists from the model table
    # silently, so this is worth failing on rather than printing past.
    if missing:
        sys.exit(f"\n{missing} site(s) still have no coordinates. Re-run to "
                 f"retry the days above; if it persists, those loc_ids are not "
                 f"appearing in any feed response and need looking at.")
