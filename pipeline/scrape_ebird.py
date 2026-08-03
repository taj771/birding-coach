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
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

KEY = os.getenv("EBIRD_API_KEY")
if not KEY:
    sys.exit("EBIRD_API_KEY not set — put it in .env locally, or in\n"
         "GitHub Settings > Secrets and variables > Actions")

REGION = os.getenv("SCRAPE_REGION", "US-PA-003")  # Allegheny County
DB = ROOT / "data" / "birding.duckdb"
DAYS = ROOT / "data" / "scrape_days.json"
DELAY = float(os.getenv("SCRAPE_DELAY", "0.4"))

DB.parent.mkdir(exist_ok=True)      # fresh checkout has no data/

BASE = "https://api.ebird.org/v2"
HEADERS = {"X-eBirdApiToken": KEY}
sess = requests.Session()

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


def get(path, **params):
    """One GET with retries. Returns parsed JSON or None."""
    for attempt in range(4):
        try:
            r = sess.get(f"{BASE}{path}", headers=HEADERS, params=params, timeout=45)
            if r.status_code == 200:
                time.sleep(DELAY)
                return r.json()
            if r.status_code == 429:          # rate limited — back off hard
                time.sleep(30 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            print(f"    HTTP {r.status_code} on {path}", flush=True)
        except requests.RequestException as e:
            print(f"    {type(e).__name__} on {path}", flush=True)
        time.sleep(3 * (attempt + 1))
    return None


def parse(c):
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
        c.get("subId"), c.get("locId"), REGION, dt, date or None, tm or None,
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


def scrape(con, dates):
    seen = {r[0] for r in con.execute("SELECT sub_id FROM checklists").fetchall()}
    print(f"already stored: {len(seen)} checklists\n", flush=True)
    calls = 0

    for i, day in enumerate(dates, 1):
        d = day["date"]
        y, m, dd = d.split("-")
        lists = get(f"/product/lists/{REGION}/{y}/{m}/{dd}", maxResults=200)
        calls += 1
        if lists is None:
            print(f"[{i}/{len(dates)}] {d}  FEED FAILED", flush=True)
            continue

        sub_ids = [x["subId"] for x in lists if x.get("subId")]
        todo = [s for s in sub_ids if s not in seen]
        print(f"[{i}/{len(dates)}] {d} ({day['kind']:>5}, wind {day['wind_max']:>4.1f}) "
              f"{len(sub_ids):>4} checklists, {len(todo):>4} new", flush=True)
        if len(sub_ids) >= 200:
            print("    WARNING: hit the 200 cap — day is truncated", flush=True)

        added = 0
        for j, sub in enumerate(todo, 1):
            c = get(f"/product/checklist/view/{sub}")
            calls += 1
            if not c:
                continue
            row, obs_rows = parse(c)
            con.execute("INSERT OR IGNORE INTO checklists VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?, now())", row)
            if obs_rows:
                con.executemany("INSERT OR IGNORE INTO observations VALUES (?,?,?)", obs_rows)
            seen.add(sub)
            added += 1
            if j % 25 == 0:
                print(f"    {j}/{len(todo)}  ({calls} calls)", flush=True)
        print(f"    +{added} stored\n", flush=True)

    return calls


if __name__ == "__main__":
    dates = json.loads(DAYS.read_text())
    print(f"region  {REGION}")
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
    calls = scrape(con, dates)

    n_ck, n_obs, n_sp, n_loc = con.execute("""
        SELECT (SELECT count(*) FROM checklists),
               (SELECT count(*) FROM observations),
               (SELECT count(DISTINCT species_code) FROM observations),
               (SELECT count(DISTINCT loc_id) FROM checklists)
    """).fetchone()

    print(f"\n=== done in {(time.time()-t0)/60:.1f} min, {calls} calls ===")
    if n_ck == 0:
        con.close()
        sys.exit("stored zero checklists. Either every requested date returned "
                 "nothing, or the feed calls failed — check the HTTP lines above.")
    print(f"  checklists   {n_ck:,}")
    print(f"  observations {n_obs:,}")
    print(f"  species      {n_sp:,}")
    print(f"  locations    {n_loc:,}")
    con.close()
