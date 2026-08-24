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
    """The TOP_HOTSPOTS busiest sites across `spec`, as {region: {loc_id}}.

    WHY NARROW AT ALL
    Scraping every hotspot in Pennsylvania for a year is about 207,000
    checklist views, and eBird answers a sustained pull with 429s carrying a
    Retry-After of six or seven seconds — so the real cost is closer to seven
    seconds a checklist than the 0.5 s SCRAPE_DELAY suggests. That is roughly
    three weeks of runner time for one state-year, which is not a thing that
    finishes.

    Birding is concentrated enough that most of it is not needed. In Allegheny
    the top 100 of 273 hotspots hold 91% of all checklists ever filed, and two
    parks alone hold 21%. Taking the head of that distribution statewide buys
    most of the record for a fraction of the calls, and the sites it keeps are
    the ones the app is asked about anyway — a hotspot with nine all-time
    checklists cannot support a site effect worth serving.

    Ranked across the whole spec rather than per county on purpose: a fixed
    number per county would spend the same budget on Forest County, four
    checklists on a good day, as on Philadelphia.

    Returns None when TOP_HOTSPOTS is unset or zero, meaning "no narrowing" —
    callers treat that as "keep everything" rather than "keep nothing".
    Regions with no selected site are absent from the mapping, which is what
    lets a caller skip their feed calls entirely.
    """
    if TOP_HOTSPOTS <= 0:
        return None

    # Keyed by the region we will actually scrape rather than by the hotspot's
    # own subnational2Code: the two normally agree, but the code is missing on
    # a few sites and a None key would quietly drop them.
    ranked = sorted(((r, h) for r in expand(spec) for h in load(r)),
                    key=lambda rh: -rh[1]["n_checklists"])

    chosen = {}
    for region, h in ranked[:TOP_HOTSPOTS]:
        chosen.setdefault(region, set()).add(h["loc_id"])
    return chosen


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
