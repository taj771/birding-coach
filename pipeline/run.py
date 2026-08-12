"""Pipeline orchestrator. One entry point, two speeds.

    python pipeline/run.py daily     scrape recent days      ~5 min
    python pipeline/run.py monthly   refit and export        ~10 min
    python pipeline/run.py all       both

Daily and monthly are separated because they change at different rates. New
checklists arrive continuously, so scraping is cheap and frequent. Model
coefficients describe how detection responds to hour, effort and weather —
that does not move week to week, so refitting monthly is enough.

Nothing here runs when a user opens the app. The app fetches its own weather
forecast and applies the stored coefficients, so the rolling seven-day window
advances on its own without the pipeline being involved.
"""
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = sys.executable

# data/ is gitignored — eBird records are not redistributable — so a fresh
# checkout has no such directory and every write below would fail
(ROOT / "data").mkdir(exist_ok=True)

# eBird submissions trail the outing — people enter a weekend trip on Monday.
# Scraping a date too soon gets only the log-in-the-field crowd, which is a
# biased sample. Seven days is enough for submissions to settle.
LAG_DAYS = 7
WINDOW_DAYS = 7


def run(script, *args):
    print(f"\n{'=' * 60}\n  {script} {' '.join(args)}\n{'=' * 60}", flush=True)
    r = subprocess.run([PY, str(ROOT / "pipeline" / script), *args])
    if r.returncode != 0:
        sys.exit(f"{script} failed with code {r.returncode}")


def write_recent_days():
    """Day list for the daily scrape: a settled window, not today."""
    import json
    end = date.today() - timedelta(days=LAG_DAYS)
    days = [{"date": str(end - timedelta(days=i)), "kind": "daily",
             "wind_max": 0.0} for i in range(WINDOW_DAYS)]
    out = ROOT / "data" / "scrape_days.json"
    out.write_text(json.dumps(sorted(days, key=lambda d: d["date"]), indent=2))
    print(f"daily window: {days[-1]['date']} .. {days[0]['date']}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    # Every mode starts from the shared history. A runner's own disk only ever
    # holds what it scraped itself — one rolling seven-day window — which is
    # far too little to fit on and would quietly publish a worse model.
    run("sync_db.py", "pull")

    if mode in ("daily", "all"):
        write_recent_days()
        run("scrape_ebird.py")          # new checklists -> duckdb
        run("backfill_locations.py")    # coordinates for any new sites
        run("sync_db.py", "push")       # hand the new days back to the others

    if mode in ("monthly", "all"):
        run("build_model_table.py")     # + weather, one row per checklist
        run("fit_logit.py")             # -> data/model_coefficients.json
        run("export_hotspots.py")       # the sites that carry their own effect
        run("fetch_photos.py")          # a picture per species per month
        run("publish.py")               # -> Supabase, or SQL to paste
        run("publish_public.py")        # -> the public dataset the app reads

    print("\ndone.")
