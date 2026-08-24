"""Scrape eBird checklists for selected days into DuckDB.

Two endpoints, as there is no bulk option:
    /product/lists/{region}/{y}/{m}/{d}   the frame  — who went birding
    /product/checklist/view/{subId}       the outcome — what they found

Idempotent: re-running skips checklists already stored, so a crash or a
Ctrl-C costs nothing. Deliberately slow — eBird's terms prohibit use that
"adversely impacts the stability of the ebird.org servers", and the bulk
path they intend for large pulls is the EBD download, not this.
"""
import json
import os
import sys
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# The key check, the session and the retry/backoff all live in ebird_api now.
# They used to live here alone, which is why backfill_locations.py had none of
# them and died on the first 429 it met.
sys.path.insert(0, str(Path(__file__).parent))
import ebird_api                                          # noqa: E402
import hotspots_ref                                       # noqa: E402
from ebird_api import get                                 # noqa: E402

# One county, or several: "US-PA-003" or "US-PA-003,US-PA-101,US-PA-029".
REGIONS = [r.strip() for r in
           os.getenv("SCRAPE_REGION", "US-PA-003").split(",") if r.strip()]

# eBird's ceiling on /product/lists. Not a number we chose, and not one the
# response admits to — a truncated day looks exactly like a day with 200
# checklists on it.
CAP = 200

# Hotspots with almost no history are not worth a call on a capped day: in
# Allegheny, dropping everything under ten all-time checklists skips 22 of 273
# sites and loses 0.09% of the county's recorded birding.
MIN_HOTSPOT_HISTORY = int(os.getenv("MIN_HOTSPOT_HISTORY", "10"))

# Keep only checklists filed at public hotspots — parks, reserves, wildlife
# areas. Personal locations are somebody's garden: eBird will not enumerate
# them, so they cannot be scraped completely on a capped day, and the app
# never forecasts for them either. Restricting the population makes the
# training data match what the app actually predicts for.
HOTSPOTS_ONLY = os.getenv("HOTSPOTS_ONLY", "0") == "1"

# The full scrape scope, which is not the same thing as the region being
# scraped right now: backfill.py works one county at a time, and ranking the
# busiest sites within a single county would give Forest County the same share
# of the budget as Philadelphia. Defaults to the region so a direct run still
# behaves sensibly.
SCOPE = os.getenv("SCRAPE_SCOPE") or ",".join(REGIONS)

# {region: {loc_id}} when TOP_HOTSPOTS narrows the scrape to the busiest
# sites, otherwise None for "every hotspot". Built once: it reads the cached
# hotspot reference, so it costs nothing per day, and holding it fixed for the
# process means every day in a run is scraped against the same site list.
SELECTED = hotspots_ref.selection(SCOPE)

DB = ROOT / "data" / "birding.duckdb"
DAYS = ROOT / "data" / "scrape_days.json"
DELAY = ebird_api.DELAY

DB.parent.mkdir(exist_ok=True)      # fresh checkout has no data/

SCHEMA = """
CREATE TABLE IF NOT EXISTS checklists (
    sub_id        VARCHAR PRIMARY KEY,
    loc_id        VARCHAR,
    region        VARCHAR,
    obs_dt        VARCHAR,
    obs_date      DATE,
    obs_time      VARCHAR,
    duration_hrs  DOUBLE,
    distance_km   DOUBLE,
    n_observers   INTEGER,
    protocol_id   VARCHAR,
    all_reported  BOOLEAN,
    richness      INTEGER,
    observer      VARCHAR,
    scraped_at    TIMESTAMP
);
CREATE TABLE IF NOT EXISTS observations (
    sub_id        VARCHAR,
    species_code  VARCHAR,
    how_many      INTEGER,
    PRIMARY KEY (sub_id, species_code)
);
"""


def day_feed(region, date):
    """Every checklist filed in a region on one day, cap worked around.

    Returns (items, calls_made, was_capped).

    The plain feed answers with at most CAP checklists and gives no hint that
    it truncated — a busy May morning and a quiet one both come back looking
    complete. Whenever it returns exactly CAP we therefore assume it did not
    finish, and ask again hotspot by hotspot.

    That works because the ceiling is per region, and a single site is nowhere
    near it: the busiest hotspot in Philadelphia on 10 May 2025 had 79
    checklists against the county's truncated 200.

    It recovers hotspot checklists completely. Personal locations cannot be
    enumerated at all, so on a capped day those stay truncated — which is the
    reason HOTSPOTS_ONLY exists.
    """
    y, m, d = str(date).split("-")
    feed = get(f"/product/lists/{region}/{y}/{m}/{d}", maxResults=CAP) or []
    if len(feed) < CAP:
        return feed, 1, False

    seen = {x["subId"]: x for x in feed if x.get("subId")}
    calls = 1
    sites = [h for h in hotspots_ref.load(region)
             if h["n_checklists"] >= MIN_HOTSPOT_HISTORY]
    # Re-asking sites we are going to discard anyway would cost one call each
    # and recover nothing: on a narrowed scrape the fan-out is the single most
    # expensive thing a day can do, and it is spent entirely on the county's
    # long tail unless it is filtered here too.
    if SELECTED is not None:
        allow = SELECTED.get(region, set())
        sites = [h for h in sites if h["loc_id"] in allow]
    print(f"    capped at {CAP} — re-asking {len(sites)} hotspots", flush=True)

    for h in sites:
        sub = get(f"/product/lists/{h['loc_id']}/{y}/{m}/{d}",
                  maxResults=CAP) or []
        calls += 1
        for x in sub:
            if x.get("subId"):
                seen.setdefault(x["subId"], x)

    print(f"    recovered {len(seen)} checklists ({len(seen) - CAP:+d}) "
          f"in {calls} calls", flush=True)
    return list(seen.values()), calls, True


def is_hotspot(item):
    """Was this checklist filed at a public hotspot?

    The feed carries the flag on the location, and it is trusted directly
    rather than checked against the cached hotspot list — a site created since
    the cache was written would otherwise be misfiled as personal.
    """
    return bool((item.get("loc") or {}).get("isHotspot"))


def parse(c, region):
    """Checklist-view JSON -> (checklist row, observation rows).

    numSpeciesAllReported comes back null, so richness is len(obs).

    `userDisplayName` is stored as an observer proxy. It is a display name
    rather than a stable ID, so it can collide or change — good enough for a
    skill covariate and for clustering standard errors, not for anything that
    needs identity.
    """
    obs = c.get("obs") or []
    dt = c.get("obsDt", "")
    date, _, tm = dt.partition(" ")
    row = (
        c.get("subId"), c.get("locId"), region, dt, date or None, tm or None,
        c.get("durationHrs"), c.get("effortDistanceKm"), c.get("numObservers"),
        c.get("protocolId"), c.get("allObsReported"), len(obs),
        c.get("userDisplayName"),
    )
    obs_rows = []
    for o in obs:
        code = o.get("speciesCode")
        if not code:
            continue
        n = o.get("howManyAtleast") or o.get("howManyAtmost")
        obs_rows.append((c.get("subId"), code, n))
    return row, obs_rows


def scrape(con, dates, regions):
    seen = {r[0] for r in con.execute("SELECT sub_id FROM checklists").fetchall()}
    print(f"already stored: {len(seen)} checklists\n", flush=True)
    calls = capped_days = 0
    units = [(r, day) for r in regions for day in dates]

    for i, (region, day) in enumerate(units, 1):
        d = day["date"]
        lists, n, was_capped = day_feed(region, d)
        calls += n
        capped_days += was_capped
        if not lists:
            print(f"[{i}/{len(units)}] {region} {d}  FEED FAILED OR EMPTY",
                  flush=True)
            continue

        if HOTSPOTS_ONLY:
            kept = [x for x in lists if is_hotspot(x)]
        else:
            kept = lists
        personal = len(lists) - len(kept)

        # The narrowing has to happen here, before the view calls: the feed
        # already told us where each checklist was filed, so a site we are not
        # keeping costs nothing to discard and roughly seven seconds to fetch.
        if SELECTED is not None:
            allow = SELECTED.get(region, set())
            before = len(kept)
            kept = [x for x in kept
                    if (x.get("loc") or {}).get("locId") in allow]
            off_list = before - len(kept)
        else:
            off_list = 0

        sub_ids = [x["subId"] for x in kept if x.get("subId")]
        todo = [s for s in sub_ids if s not in seen]
        drop = f", {personal:>3} personal skipped" if HOTSPOTS_ONLY else ""
        drop += f", {off_list:>4} off-list skipped" if SELECTED is not None else ""
        print(f"[{i}/{len(units)}] {region} {d} ({day['kind']:>8}) "
              f"{len(sub_ids):>4} checklists, {len(todo):>4} new{drop}",
              flush=True)

        added = 0
        for j, sub in enumerate(todo, 1):
            c = get(f"/product/checklist/view/{sub}")
            calls += 1
            if not c:
                continue
            row, obs_rows = parse(c, region)
            # Columns are named, not positional. `observer` was added by ALTER
            # TABLE after the first scrapes, so it sits at the end of an older
            # database and in the middle of one built from CREATE TABLE. A
            # positional insert silently wrote a birder's name into scraped_at
            # on whichever of the two it was not written for.
            con.execute(
                "INSERT OR IGNORE INTO checklists "
                "(sub_id, loc_id, region, obs_dt, obs_date, obs_time, "
                " duration_hrs, distance_km, n_observers, protocol_id, "
                " all_reported, richness, observer, scraped_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, now())", row)
            if obs_rows:
                con.executemany(
                    "INSERT OR IGNORE INTO observations "
                    "(sub_id, species_code, how_many) VALUES (?,?,?)", obs_rows)
            seen.add(sub)
            added += 1
            if j % 25 == 0:
                print(f"    {j}/{len(todo)}  ({calls} calls)", flush=True)
        print(f"    +{added} stored\n", flush=True)

    return calls, capped_days


if __name__ == "__main__":
    dates = json.loads(DAYS.read_text())
    print(f"regions {len(REGIONS)}: {', '.join(REGIONS[:6])}{' ...' if len(REGIONS) > 6 else ''}")
    print(f"scope   {'hotspots only' if HOTSPOTS_ONLY else 'all locations'}")
    if SELECTED is not None:
        n_sites = sum(len(v) for v in SELECTED.values())
        mine = len(SELECTED.get(REGIONS[0], set())) if len(REGIONS) == 1 else None
        print(f"sites   top {hotspots_ref.TOP_HOTSPOTS} across {SCOPE} "
              f"({n_sites:,} selected in {len(SELECTED)} counties"
              f"{f', {mine} here' if mine is not None else ''})")
    print(f"days    {len(dates)}  ({dates[0]['date']} .. {dates[-1]['date']})")
    print(f"db      {DB}")
    print(f"delay   {DELAY}s\n", flush=True)

    con = duckdb.connect(str(DB))
    con.execute(SCHEMA)
    # migrate older databases created before the observer column existed
    try:
        con.execute("ALTER TABLE checklists ADD COLUMN observer VARCHAR")
        print("added `observer` column to existing table", flush=True)
    except duckdb.Error:
        pass
    t0 = time.time()
    calls, capped_days = scrape(con, dates, REGIONS)

    n_ck, n_obs, n_sp, n_loc = con.execute("""
        SELECT (SELECT count(*) FROM checklists),
               (SELECT count(*) FROM observations),
               (SELECT count(DISTINCT species_code) FROM observations),
               (SELECT count(DISTINCT loc_id) FROM checklists)
    """).fetchone()

    print(f"\n=== done in {(time.time()-t0)/60:.1f} min, {calls} calls ===")
    # Worth stating rather than leaving in the scroll-back: a capped day is one
    # the plain feed would have silently truncated, and the count is the
    # clearest signal of how much the hotspot fallback is earning.
    print(f"  capped days  {capped_days} (re-asked hotspot by hotspot)")
    if n_ck == 0:
        con.close()
        sys.exit("stored zero checklists. Either every requested date returned "
                 "nothing, or the feed calls failed — check the HTTP lines above.")
    print(f"  checklists   {n_ck:,}")
    print(f"  observations {n_obs:,}")
    print(f"  species      {n_sp:,}")
    print(f"  locations    {n_loc:,}")
    con.close()
