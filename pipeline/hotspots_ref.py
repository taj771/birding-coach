"""Hotspot lists per county, fetched once and cached.

    python pipeline/hotspots_ref.py US-PA-003        one county
    python pipeline/hotspots_ref.py US-PA            every county in a state

WHY THIS EXISTS
The day feed caps at 200 checklists per region-day. A county busts that ceiling
on migration days — Philadelphia, Allegheny and Chester all returned exactly
200 on 10 May 2025, with no way to know the true total. A single hotspot never
comes close: the busiest site in Philadelphia that day had 79.

So a capped county-day is re-asked hotspot by hotspot, and that needs a list of
the county's hotspots. It changes about as often as parks get built, so it is
fetched once and cached rather than requested per run.

ORDERING
Sorted by numChecklistsAllTime, descending — eBird's own count of how many
checklists have ever been filed there. Birding is extremely concentrated: in
Allegheny the top 100 of 273 hotspots hold 91% of all checklists, and North
Park and Frick Park alone are 21%. Descending order is what lets the caller
stop early once the tail stops returning anything.

Note this is *not* numSpeciesAllTime, which is species richness. A quiet marsh
can out-rank a busy city park on species and be nearly empty of checklists.
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(Path(__file__).parent))
import ebird_api                                          # noqa: E402

CACHE = ROOT / "data" / "hotspots_ref"

# How many of the busiest sites to scrape, across the whole scrape scope.
# Zero means every hotspot, which is the behaviour this had before the
# narrowing existed — see selection() for why a cap is wanted at all. Read
# here rather than in each caller so the scraper and the work-queue planner
# cannot disagree about which sites are in scope.
TOP_HOTSPOTS = int(os.getenv("TOP_HOTSPOTS", "0"))
# Per county instead of across the state, for coverage rather than volume, and
# a cap on how many counties are worth visiting at all. See selection().
PER_COUNTY_HOTSPOTS = int(os.getenv("PER_COUNTY_HOTSPOTS", "0"))
MAX_COUNTIES = int(os.getenv("MAX_COUNTIES", "0"))


def fetch(region):
    """eBird's hotspot list for a region, busiest first."""
    hs = ebird_api.get(f"/ref/hotspot/{region}", fmt="json")
    if hs is None:
        return None
    hs.sort(key=lambda h: -(h.get("numChecklistsAllTime") or 0))
    return [{"loc_id": h["locId"],
             "name": h.get("locName", ""),
             "lat": h.get("lat"),
             "lon": h.get("lng"),
             "county": h.get("subnational2Code"),
             "n_checklists": h.get("numChecklistsAllTime") or 0,
             "latest_obs": h.get("latestObsDt")}
            for h in hs]


def split_state(state, refresh=False):
    """Fetch a whole state once and write one cache file per county.

    /ref/hotspot/US-PA returns all 4,941 Pennsylvania hotspots in a single
    response, each carrying its subnational2Code. Asking county by county
    would be 67 calls for the same bytes, so the state call is split locally
    instead. Counties with no hotspots simply get no file.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    hs = fetch(state)
    if hs is None:
        return 0
    by_county = {}
    for h in hs:
        if h["county"]:
            by_county.setdefault(h["county"], []).append(h)
    for county, rows in by_county.items():
        (CACHE / f"{county}.json").write_text(
            json.dumps(rows, separators=(",", ":")))
    return len(by_county)


def load(region, refresh=False):
    """Cached hotspot list for one county. Fetches on first use."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{region}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    # A county cache miss usually means the state has never been fetched, and
    # fetching the state is the same single call that fills every other county
    # too. Falling back to the state avoids 67 separate requests the first
    # time a statewide scrape starts.
    state = "-".join(region.split("-")[:2])
    if region != state and split_state(state):
        if path.exists():
            return json.loads(path.read_text())

    hs = fetch(region)
    if hs is None:
        # Returning empty rather than raising: a county whose hotspot list we
        # cannot fetch should degrade to the plain county feed, not stop a
        # scrape that has other counties to get through.
        print(f"  {region}: hotspot list unavailable", flush=True)
        return []
    path.write_text(json.dumps(hs, separators=(",", ":")))
    return hs


def ids(region):
    """Just the locIds, as a set — for deciding whether a checklist counts."""
    return {h["loc_id"] for h in load(region)}


def expand(spec):
    """'US-PA' -> its counties; 'US-PA-003,US-PA-101' -> that list.

    A state code has one hyphen, a county code two, so which was meant can be
    read off the string rather than asked for separately.
    """
    out = []
    for p in (s.strip() for s in spec.split(",")):
        if p:
            out.extend(counties(p) if p.count("-") == 1 else [p])
    return out


def selection(spec):
    """Which sites are worth scraping in `spec`, as {region: {loc_id}}.

    WHY NARROW AT ALL
    Scraping every hotspot in Pennsylvania for a year is about 207,000
    checklist views, and eBird answers a sustained pull with 429s carrying a
    Retry-After of six or seven seconds — so the real cost is closer to seven
    seconds a checklist than the 0.5 s SCRAPE_DELAY suggests. That is roughly
    three weeks of runner time for one state-year, which is not a thing that
    finishes.

    Birding is concentrated enough that most of it is not needed. In Allegheny
    the top 100 of 273 hotspots hold 91% of all checklists ever filed, and two
    parks alone hold 21%.

    THREE KNOBS, BECAUSE "NARROW" MEANS TWO DIFFERENT THINGS
    TOP_HOTSPOTS ranks sites across the whole spec and keeps the busiest N.
    That buys the most records per call, and it is the right answer when the
    model is what matters — but it concentrates. Pennsylvania's birding sits
    in the southeast, so a statewide top 400 leaves most counties with nothing
    at all and a map of the state looking empty outside Philadelphia.

    PER_COUNTY_HOTSPOTS instead keeps each county's own busiest N. It costs
    more per record — a quiet county's tenth site is worth less than a busy
    county's hundredth — and it keeps every county in the work queue, which is
    one feed call per county per day whether or not the day yields anything.
    What it buys is coverage everywhere, which is what a demo needs: somewhere
    to point at in any county someone asks about.

    MAX_COUNTIES trims the tail of counties entirely, for when a handful of
    near-empty ones are not worth their feed calls.

    They compose, applied in that order: counties are trimmed, then each
    surviving county is trimmed, then an overall cap is applied to whatever is
    left. Each is off at zero, and with all three off this returns None,
    meaning "no narrowing" — callers read that as keep everything rather than
    keep nothing. Regions with no selected site are absent from the mapping,
    which is what lets a caller skip their feed calls entirely.
    """
    if TOP_HOTSPOTS <= 0 and PER_COUNTY_HOTSPOTS <= 0 and MAX_COUNTIES <= 0:
        return None

    # Keyed by the region we will actually scrape rather than by the hotspot's
    # own subnational2Code: the two normally agree, but the code is missing on
    # a few sites and a None key would quietly drop them.
    by_region = {r: sorted(load(r), key=lambda h: -h["n_checklists"])
                 for r in expand(spec)}

    if MAX_COUNTIES > 0:
        busiest = sorted(by_region,
                         key=lambda r: -sum(h["n_checklists"]
                                            for h in by_region[r]))
        by_region = {r: by_region[r] for r in busiest[:MAX_COUNTIES]}

    if PER_COUNTY_HOTSPOTS > 0:
        by_region = {r: hs[:PER_COUNTY_HOTSPOTS] for r, hs in by_region.items()}

    if TOP_HOTSPOTS > 0:
        ranked = sorted(((r, h) for r, hs in by_region.items() for h in hs),
                        key=lambda rh: -rh[1]["n_checklists"])[:TOP_HOTSPOTS]
        chosen = {}
        for region, h in ranked:
            chosen.setdefault(region, set()).add(h["loc_id"])
        return chosen

    return {r: {h["loc_id"] for h in hs} for r, hs in by_region.items() if hs}


def counties(state):
    """County codes for a state, e.g. US-PA -> ['US-PA-001', ...].

    Answered from the cache when it is populated. split_state() writes one
    file per county and the filename is the code, so once a state has been
    fetched this needs no request at all — which also means a backfill can
    build its work queue with the network unavailable.
    """
    cached = sorted(p.stem for p in CACHE.glob(f"{state}-*.json"))
    if cached:
        return cached

    r = ebird_api.get(f"/ref/region/list/subnational2/{state}")
    if r is not None:
        return [c["code"] for c in r]

    # Last resort: fetching the state's hotspots also tells us its counties,
    # and it is the call we would be making shortly anyway.
    if split_state(state):
        return sorted(p.stem for p in CACHE.glob(f"{state}-*.json"))
    sys.exit(f"could not list counties for {state}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("\n\n")[0] + "\n\n"
                 "usage: hotspots_ref.py <region>   e.g. US-PA-003 or US-PA")

    target = sys.argv[1]
    regions = counties(target) if target.count("-") == 1 else [target]
    print(f"{len(regions)} region(s)\n")

    total = 0
    for i, region in enumerate(regions, 1):
        hs = load(region, refresh="--refresh" in sys.argv)
        total += len(hs)
        top = hs[0]["name"][:44] if hs else "-"
        print(f"[{i}/{len(regions)}] {region}  {len(hs):>4} hotspots   "
              f"busiest: {top}", flush=True)

    print(f"\n{total:,} hotspots cached in {CACHE}")
