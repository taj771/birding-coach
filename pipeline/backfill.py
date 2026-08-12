"""Scrape a date range to fill gaps in the calendar.

    python pipeline/backfill.py 2025-08-01 2025-11-01
    python pipeline/backfill.py 2025-08-01 2025-11-01 --max-days 30

WHY THIS EXISTS
The daily job scrapes a rolling seven-day window, so the database grows
forwards one day at a time and the calendar fills at the speed of real time.
The model needs the opposite: coverage across the whole year, now.

Week terms are indexed by day-of-year, not by date, so a gap at week 31 is
filled by scraping *any* year's week 31. Last year's August is as good as this
year's for that purpose and is already settled, which is why the range below is
historical rather than recent.

RESUMABLE
Days already in the database are skipped, so re-running costs nothing and an
interrupted run continues where it stopped. `--max-days` caps one run, which is
how a long backfill is split across several CI jobs without hitting the job
timeout.

COST
Two eBird calls per checklist, plus one per day for the feed. Allegheny averages
about 90 complete checklists a day outside migration and about 150 in May, so a
three-month backfill is roughly 9,000 calls — around an hour at the default
delay. The estimate is printed before anything is fetched.
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
PY = sys.executable
DB = ROOT / "data" / "birding.duckdb"
DAYS_FILE = ROOT / "data" / "scrape_days.json"

# eBird submissions trail the outing; the same seven days the daily job waits
LAG_DAYS = 7
# rough checklists per day in Allegheny, for the estimate only
PER_DAY = 110
# Days per push. A run cancelled at day 34 of a 40-day chunk lost all 34,
# because nothing had been pushed yet — so this is now small enough that a
# cancellation costs minutes rather than an hour, while still not shipping the
# whole database to Hugging Face every ninety seconds.
CHUNK = 15


def run(script, *args):
    print(f"\n{'=' * 60}\n  {script} {' '.join(args)}\n{'=' * 60}", flush=True)
    r = subprocess.run([PY, str(ROOT / "pipeline" / script), *args])
    if r.returncode != 0:
        sys.exit(f"{script} failed with code {r.returncode}")


def already_have():
    if not DB.exists():
        return set()
    con = duckdb.connect(str(DB), read_only=True)
    have = {r[0] for r in con.execute(
        "select distinct obs_date from checklists").fetchall()}
    con.close()
    return have


def parse(s):
    # stripped first: a date typed into a web form arrives with whatever
    # whitespace came with it, and refusing a trailing space helps nobody
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError:
        sys.exit(f"dates must look like 2025-08-01, got {s!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("start", help="first date, e.g. 2025-08-01")
    p.add_argument("end", help="last date, inclusive")
    p.add_argument("--max-days", type=int, default=None,
                   help="stop after this many days; re-run to continue")
    a = p.parse_args()

    start, end, cap = parse(a.start), parse(a.end), a.max_days
    if start > end:
        sys.exit("start date is after end date")

    settled = date.today() - timedelta(days=LAG_DAYS)
    if end > settled:
        print(f"end date trimmed to {settled}: eBird submissions trail the "
              f"outing by about a week, and scraping fresher days would "
              f"sample only the log-in-the-field crowd")
        end = settled

    (ROOT / "data").mkdir(exist_ok=True)
    have = already_have()
    wanted = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in wanted if d not in have]

    print(f"range      {start} .. {end}  ({len(wanted)} days)")
    print(f"already in {len(wanted) - len(todo)} days")
    print(f"to scrape  {len(todo)} days")
    if cap and len(todo) > cap:
        todo = todo[:cap]
        print(f"capped at  {cap} this run — re-run to continue")
    if not todo:
        print("\nnothing to do; the range is already covered.")
        raise SystemExit(0)

    calls = len(todo) * (PER_DAY + 1)
    print(f"\nestimate   ~{calls:,} eBird calls, "
          f"~{calls * 0.4 / 3600:.1f} h at 0.4 s between them")
    print(f"weeks hit  {sorted({d.timetuple().tm_yday // 7 for d in todo})}")

    # Scrape in chunks and push after each one. A long backfill can outlast a
    # CI job's timeout, and a single push at the end would mean a killed job
    # threw away everything it had fetched. Pushing every CHUNK days bounds the
    # loss to one chunk, and re-running resumes from whatever landed.
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        print(f"\n{'#' * 60}\n#  chunk {i // CHUNK + 1}: {batch[0]} .. {batch[-1]} "
              f"({len(batch)} days)\n{'#' * 60}", flush=True)
        DAYS_FILE.write_text(json.dumps(
            [{"date": str(d), "kind": "backfill", "wind_max": 0.0} for d in batch],
            indent=2))
        run("scrape_ebird.py")
        run("backfill_locations.py")
        run("sync_db.py", "push")

    left = [d for d in wanted if d not in already_have()]
    print(f"\ndone. {len(left)} days of the range still missing.")
    if left:
        print(f"re-run the same command to continue from {left[0]}.")
