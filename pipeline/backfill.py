"""Scrape a date range across one or many regions, to fill gaps in the calendar.

    python pipeline/backfill.py 2025-08-01 2026-07-31
    python pipeline/backfill.py 2025-08-01 2026-07-31 --regions US-PA
    python pipeline/backfill.py 2025-08-01 2026-07-31 --budget-min 300

WHY THIS EXISTS
The daily job scrapes a rolling seven-day window, so the database grows
forwards one day at a time and the calendar fills at the speed of real time.
The model needs the opposite: coverage across the whole year, now.

Week terms are indexed by day-of-year, not by date, so a gap at week 31 is
filled by scraping *any* year's week 31. Last year's August is as good as this
year's for that purpose and is already settled.

THE UNIT OF WORK IS A REGION-DAY
A single county was one dimension, so a day was a work unit. Statewide it is a
pair: Allegheny on 3 May is a different unit from Chester on 3 May. Passing a
state code expands it to that state's counties.

Counties are ordered by how much birding happens in them, busiest first. A run
that stops half way then leaves the high-traffic counties complete rather than
sixty-seven counties each a third done — the first is a usable model, the
second is not.

RESUMABLE
Region-days already in the database are skipped, so re-running costs nothing
and an interrupted run continues where it stopped. That is a query rather than
a bookmark, so it survives a job being killed, a cache being evicted, or a run
that never started.

TIME BUDGET
A GitHub job is killed at six hours; ours stops itself at five. Being killed
loses whatever the current chunk had fetched, because the push had not
happened yet. Stopping deliberately finishes the chunk, pushes it, and exits
green — which matters when the run is one of nine.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent.parent
PY = sys.executable
DB = ROOT / "data" / "birding.duckdb"
DAYS_FILE = ROOT / "data" / "scrape_days.json"

sys.path.insert(0, str(Path(__file__).parent))

# eBird submissions trail the outing; the same seven days the daily job waits
LAG_DAYS = 7
# Hotspot checklists filed per day across the whole state, for the estimate
# only. Derived from measurement rather than guessed: Allegheny yields about 42
# hotspot checklists a day and holds 7.2% of Pennsylvania's all-time hotspot
# checklists, so the state runs at roughly 42 / 0.072.
STATE_PER_DAY = float(os.getenv("STATE_PER_DAY", "580"))
# Region-days per push. A run cancelled at day 34 of a 40-day chunk lost all
# 34, because nothing had been pushed yet — so this is small enough that a
# cancellation costs minutes rather than an hour, while still not shipping the
# whole database to Hugging Face every ninety seconds.
CHUNK = int(os.getenv("BACKFILL_CHUNK", "15"))
# Seconds per call for the estimate, and NOT the same thing as SCRAPE_DELAY.
# The delay is what we wait between calls; this is what a call actually costs
# once eBird starts refusing them. A statewide run on 2026-08-24 took 329
# minutes to fetch about 2,700 checklists — 2,632 of those calls came back 429
# with a Retry-After of six or seven seconds, so the sustained rate was 7.3 s
# per checklist against the 0.5 s the delay implies. Estimating at the delay
# under-priced a full state-year by fourteen times, which is the difference
# between a two-day backfill and a three-week one.
SEC_PER_CALL = float(os.getenv("SEC_PER_CALL", "7.3"))
# Stop this many minutes in. The job's own ceiling is 330; leaving half an hour
# means the last chunk and its push finish inside it.
BUDGET_MIN = int(os.getenv("BUDGET_MIN", "300"))
# The wall the runner enforces, not one we choose: the job's timeout-minutes is
# 330 and being killed there loses the current chunk. Kept below it with enough
# room for the final push and the summary.
HARD_MIN = int(os.getenv("HARD_MIN", "315"))


def run(script, *args, env=None):
    print(f"\n{'=' * 60}\n  {script} {' '.join(args)}\n{'=' * 60}", flush=True)
    r = subprocess.run([PY, str(ROOT / "pipeline" / script), *args],
                       env={**os.environ, **(env or {})})
    if r.returncode != 0:
        sys.exit(f"{script} failed with code {r.returncode}")


def sample_days(start, end, every):
    """Dates from start to end, one per `every`-day block. every=1 is all.

    WHY SAMPLE AT ALL
    Week enters the model as a categorical — fit_logit does `week = doy // 7`
    and fits a coefficient per week — so a week with no checklists has no
    coefficient and cannot be forecast at all. There is no borrowing from
    neighbouring weeks. That makes a thin sample spread over the whole year
    worth far more than a dense block of it: three contiguous months would
    leave forty of the fifty-three week cells permanently unfittable.

    WHY IT GROUPS BY THE MODEL'S OWN WEEK
    The obvious implementation — every Nth day from the start date — fails
    twice. It lands on the same weekday all year, and birding is emphatically
    not weekday-invariant: weekend mornings carry far more checklists, by
    different people, walking further. Train on Sundays, get asked about
    Tuesdays.

    Rotating the offset fixes the weekday but introduces a subtler bug. An
    effective stride of eight days drifts against `doy // 7`, whose boundaries
    are fixed by the calendar, so roughly one week in eight gets no sample at
    all while its neighbour gets two. Measured over a year: 44 of 53 week
    cells covered. Nine unfittable weeks, from a routine that exists purely to
    make every week fittable.

    So group by the week index the model actually uses and take one day from
    each, stepping which day through the group. Full week coverage by
    construction rather than by arithmetic that happens to work out.
    """
    if every < 1:
        sys.exit(f"--every must be at least 1, got {every}")
    days = [start + timedelta(days=i)
            for i in range((end - start).days + 1)]
    if every == 1:
        return days
    if every < 7:
        # Below a week there is at least one sample per week whatever the
        # phase, so plain striding is safe — and 2..6 are all coprime with 7,
        # which walks the weekday round on its own.
        return days[::every]

    by_week = {}
    for d in days:
        by_week.setdefault((d.year, d.timetuple().tm_yday // 7), []).append(d)

    # WHY THE PICK IS A FUNCTION OF THE WEEK AND NOT OF ITS POSITION
    # A scheduled backfill has no start date of its own: it asks for the last
    # twelve months, so `start` moves forward every single day. Choosing the
    # day by how far through the range its week sits meant that shifting the
    # range by one day re-picked every week. Measured: three days of drift left
    # one day in common out of fifty-three, and seven days left none at all.
    #
    # Which does not fail loudly. already_have() would find the newly chosen
    # days unscraped and scrape them, so the job would keep working, keep
    # pushing, and never converge — outstanding would fall all day and be back
    # up by morning. Anchoring the choice to the calendar week itself makes the
    # sample the same set whatever start and end happen to be.
    step = max(1, round(every / 7))
    return [ds[(year * 53 + week) % len(ds)]
            for (year, week), ds in sorted(by_week.items())[::step]]


def stored_by_region():
    """[(region, days, checklists)] already in the database, busiest first.

    What a run is adding to is not obvious from the outside: data/ is
    gitignored, the database lives in a private dataset, and the counts in a
    push message are totals with no breakdown. Printing this before a plan
    makes "how much of the state is actually covered" answerable without
    downloading twenty megabytes and opening DuckDB by hand.
    """
    if not DB.exists() or DB.stat().st_size == 0:
        return []
    con = duckdb.connect(str(DB), read_only=True)
    try:
        rows = con.execute("""
            select region, count(distinct obs_date), count(*)
            from checklists group by region order by 3 desc""").fetchall()
    except duckdb.Error:
        rows = []
    con.close()
    return rows


def already_have():
    """{(region, date)} already scraped."""
    if not DB.exists() or DB.stat().st_size == 0:
        return set()
    con = duckdb.connect(str(DB), read_only=True)
    try:
        rows = con.execute(
            "select distinct region, obs_date from checklists").fetchall()
    except duckdb.Error:
        rows = []
    con.close()
    return {(r[0], r[1]) for r in rows}


def expand(spec):
    """'US-PA' -> its counties; 'US-PA-003,US-PA-101' -> that list.

    Ordered busiest first, using eBird's own all-time checklist counts per
    hotspot summed by county. A partial run then leaves whole high-traffic
    counties finished instead of every county part-done.

    When TOP_HOTSPOTS narrows the scrape, counties holding none of the chosen
    sites are dropped from the queue entirely rather than left in it to
    contribute a feed call a day that can only be discarded — across a year
    that alone is one call per county per day for counties we would keep
    nothing from. Volume is then counted over the selected sites only, so the
    ordering and the estimate both describe the work actually queued.
    """
    import hotspots_ref

    regions = hotspots_ref.expand(spec)
    selected = hotspots_ref.selection(spec)
    if selected is not None:
        regions = [r for r in regions if selected.get(r)]

    volume = {}
    for r in regions:
        keep = selected.get(r) if selected is not None else None
        volume[r] = sum(h["n_checklists"] for h in hotspots_ref.load(r)
                        if keep is None or h["loc_id"] in keep)

    # The denominator for the estimate is every hotspot in the spec, narrowed
    # or not, because STATE_PER_DAY was measured against the unnarrowed state.
    # Normalising against the selected total instead would re-inflate the
    # figure to the full state's daily volume and hide the whole saving.
    everything = sum(sum(h["n_checklists"] for h in hotspots_ref.load(r))
                     for r in hotspots_ref.expand(spec))
    return sorted(regions, key=lambda r: -volume.get(r, 0)), volume, everything


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
    p.add_argument("--regions", default=os.getenv("SCRAPE_REGION", "US-PA-003"),
                   help="county codes, or a state code to expand")
    p.add_argument("--max-units", type=int, default=None,
                   help="stop after this many region-days; re-run to continue")
    p.add_argument("--budget-min", type=int, default=BUDGET_MIN,
                   help="stop cleanly after this many minutes")
    # A region-day is never revisited once it holds any checklist, so the site
    # selection is effectively permanent for every day it touches: widening it
    # later leaves the days already scraped as they were. That makes "what
    # would this actually fetch" a question worth answering before a multi-day
    # run rather than during one.
    p.add_argument("--plan", action="store_true",
                   help="print the plan and the estimate, then exit")
    p.add_argument("--every", type=int, default=int(os.getenv("EVERY", "1")),
                   help="sample one day per N (7 = weekly); 1 scrapes them all")
    a = p.parse_args()

    t0 = time.time()
    start, end = parse(a.start), parse(a.end)
    if start > end:
        sys.exit("start date is after end date")

    settled = date.today() - timedelta(days=LAG_DAYS)
    if end > settled:
        print(f"end date trimmed to {settled}: eBird submissions trail the "
              f"outing by about a week, and scraping fresher days would "
              f"sample only the log-in-the-field crowd")
        end = settled

    (ROOT / "data").mkdir(exist_ok=True)

    # Pull before anything reads the database, for two separate reasons.
    #
    # The dangerous one: sync_db.py push uploads the local file whole, it does
    # not merge. run.py pulls first so its push can only ever be a superset,
    # but the workflow calls this script directly and it did not, which left
    # the entire history depending on the Actions cache restoring data/. On a
    # cache miss — evicted after seven days, or under the repo cache limit —
    # this would have scraped fifteen region-days into an empty database and
    # published that over everything. Nothing had been lost yet; the cache had
    # been restoring. It was one eviction away.
    #
    # The ordinary one: already_have() is what makes a run resumable, and it
    # can only skip the region-days it can see. A runner that starts blind
    # re-scrapes days another machine already paid for.
    #
    # --plan pulls too. It reports what is outstanding, which is a statement
    # about the stored database and not about this runner's disk — answering
    # it from whatever the cache happened to restore would overstate the work
    # left, which is the one number somebody runs a plan to find out.
    run("sync_db.py", "pull")

    regions, volume, all_volume = expand(a.regions)
    days = sample_days(start, end, a.every)
    have = already_have()

    # region-major: finish a county before starting the next one
    todo = [(r, d) for r in regions for d in days if (r, d) not in have]

    import hotspots_ref
    selected = hotspots_ref.selection(a.regions)
    if selected is not None:
        n_sites = sum(len(v) for v in selected.values())
        share = (sum(volume.values()) / all_volume) if all_volume else 0
        rule = []
        if hotspots_ref.MAX_COUNTIES > 0:
            rule.append(f"busiest {hotspots_ref.MAX_COUNTIES} counties")
        if hotspots_ref.PER_COUNTY_HOTSPOTS > 0:
            rule.append(f"top {hotspots_ref.PER_COUNTY_HOTSPOTS} per county")
        if hotspots_ref.TOP_HOTSPOTS > 0:
            rule.append(f"top {hotspots_ref.TOP_HOTSPOTS} overall")
        print(f"sites      {n_sites:,} hotspots ({' + '.join(rule)}) "
              f"across {len(regions)} of "
              f"{len(hotspots_ref.expand(a.regions))} counties")
        # The share is the honest headline: it is how much of the region's
        # recorded birding these sites actually hold, and a coverage-first
        # selection buys a much smaller one than a volume-first selection of
        # the same size. Worth seeing before a multi-day run, not after.
        print(f"           {share:.0%} of all recorded birding in {a.regions}")

    print(f"regions    {len(regions)}  ({', '.join(regions[:4])}"
          f"{' ...' if len(regions) > 4 else ''})")
    print(f"range      {start} .. {end}  ({len(days)} days)")
    print(f"units      {len(regions) * len(days):,} region-days, "
          f"{len(todo):,} outstanding")
    if a.max_units and len(todo) > a.max_units:
        todo = todo[:a.max_units]
        print(f"capped at  {a.max_units:,} this run")
    if not todo:
        print("\nnothing to do; the range is already covered.")
        raise SystemExit(0)

    # Estimate per county rather than per unit. A flat rate would price Forest
    # County — four checklists on a good day — the same as Allegheny, and
    # across sixty-seven counties that is not a rounding error: it inflates the
    # figure roughly eightfold, which is the difference between "leave it
    # running for two days" and "this is not worth doing".
    #
    # Each county's share of the state's all-time hotspot checklists is used as
    # its share of current activity, scaled by the rate we actually measured in
    # Allegheny: 42 hotspot checklists a day at 7.2% of the state's volume.
    #
    # Divided by every hotspot in the spec rather than by the selected ones, so
    # that narrowing the site list shows up here as a smaller number instead of
    # being normalised back out.
    per_day = {r: max(1.0, (volume.get(r, 0) / (all_volume or 1)) * STATE_PER_DAY)
               for r in regions}
    views = sum(per_day[r] for r, _ in todo)
    calls = len(todo) + views          # one feed per unit, one view per checklist
    print(f"estimate   ~{calls:,.0f} eBird calls "
          f"({len(todo):,} feeds + ~{views:,.0f} checklists), "
          f"~{calls * SEC_PER_CALL / 3600:.0f} h at {SEC_PER_CALL} s a call")
    print(f"           excludes hotspot fan-out on capped days")
    print(f"budget     {a.budget_min} min this run")
    print(f"           ~{calls * SEC_PER_CALL / 3600 / (a.budget_min / 60):.0f} "
          f"runs of {a.budget_min} min to finish", flush=True)

    if a.plan:
        have_rows = stored_by_region()
        if have_rows:
            tot_d = sum(r[1] for r in have_rows)
            tot_c = sum(r[2] for r in have_rows)
            print(f"\nalready stored: {tot_c:,} checklists over {tot_d:,} "
                  f"region-days in {len(have_rows)} region(s)")
            for region, days_n, n in have_rows[:15]:
                print(f"  {region:<12} {days_n:>4} days  {n:>7,} checklists")
            if len(have_rows) > 15:
                print(f"  ... and {len(have_rows) - 15} more")
        else:
            print("\nalready stored: nothing — the database is empty.")
        print("\n--plan: nothing scraped.")
        raise SystemExit(0)
    print(flush=True)

    done = stopped = 0
    for i in range(0, len(todo), CHUNK):
        spent = (time.time() - t0) / 60
        batch = todo[i:i + CHUNK]

        # Two ways to decide to stop, and the second one is why this is not
        # just the budget check.
        #
        # The budget is only consulted here, between chunks, and a chunk takes
        # over an hour. So a chunk starting at minute 299 of a 300-minute
        # budget runs to minute 380 — against a job the runner kills at 330,
        # losing everything that chunk had fetched since the last push. Both
        # runs on 2026-08-24 survived that on luck: one stopped at 324 min and
        # finished at 329, a single minute inside the ceiling.
        #
        # So refuse to *start* a chunk that cannot finish. The rate comes from
        # this run rather than a constant, because it varies by county — a
        # chunk of Philadelphia days is not a chunk of Forest County days.
        why = None
        if spent > a.budget_min:
            why = f"time budget reached at {spent:.0f} min"
        elif done and spent + (spent / done) * len(batch) > HARD_MIN:
            why = (f"next chunk would run past {HARD_MIN} min "
                   f"(at {spent:.0f} min, ~{(spent / done) * len(batch):.0f} "
                   f"min for {len(batch)} more)")
        if why:
            # Deliberate stop rather than being killed mid-chunk: everything
            # up to here has been pushed, and the next run picks the rest up
            # from the database instead of from a plan it has to remember.
            print(f"\n{'#' * 60}\n#  {why} — stopping cleanly\n{'#' * 60}",
                  flush=True)
            stopped = 1
            break

        # A chunk is one region at a time: scrape_ebird takes SCRAPE_REGION,
        # and mixing counties inside a chunk would mean the day list no longer
        # describes what is being fetched.
        by_region = {}
        for r, d in batch:
            by_region.setdefault(r, []).append(d)

        for region, dates in by_region.items():
            print(f"\n{'#' * 60}\n#  {region}: {dates[0]} .. {dates[-1]} "
                  f"({len(dates)} days)   [{done:,}/{len(todo):,} done, "
                  f"{spent:.0f} min]\n{'#' * 60}", flush=True)
            DAYS_FILE.write_text(json.dumps(
                [{"date": str(d), "kind": "backfill", "wind_max": 0.0}
                 for d in dates], indent=2))
            try:
                # SCRAPE_SCOPE is the whole spec, not this county: the site
                # ranking has to be the same one the work queue was built
                # from, and a child that only saw its own county would rank
                # within it and keep a different set of sites each chunk.
                env = {"SCRAPE_REGION": region, "SCRAPE_SCOPE": a.regions}
                run("scrape_ebird.py", env=env)
                run("backfill_locations.py", env=env)
            finally:
                run("sync_db.py", "push")
            done += len(dates)

    left = [(r, d) for r in regions for d in days
            if (r, d) not in already_have()]
    mins = (time.time() - t0) / 60
    print(f"\n{'=' * 60}")
    print(f"scraped {done:,} region-days in {mins:.0f} min")
    print(f"{len(left):,} of {len(regions) * len(days):,} still outstanding")
    if left:
        nxt = left[0]
        print(f"re-run the same command to continue from {nxt[0]} {nxt[1]}.")
    if stopped:
        print("stopped on the time budget, not an error — this run is green.")
