"""Resolve a photo of each species, for each month, from iNaturalist.

    python pipeline/fetch_photos.py        -> data/photos.json

WHY PER MONTH
A single photo of a bird is nearly always a breeding male, and the birds a
beginner cannot name are precisely the ones that do not look like that: a female
cardinal is fawn, autumn warblers are drab, juveniles look like nothing in
particular. Showing a May photo to someone birding in September is worse than
showing nothing, because it produces confident wrong answers.

iNaturalist can filter observations by the month they were taken, so the picker
can show what people actually photographed in the month being logged. Where a
species is sexually dimorphic a female photo is fetched too.

WHY NOT AT REQUEST TIME
Resolved here and published in the bundle: one place handling attribution, no
runtime dependency on a third-party API, and no rate limit in front of a user.

CACHING
Existing entries are kept and only missing ones fetched, so a re-run after a
refit costs a handful of requests rather than nine hundred.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
COEF = ROOT / "data" / "model_coefficients.json"
OUT = ROOT / "data" / "photos.json"

INAT = "https://api.inaturalist.org/v1"
PAUSE = 0.6          # iNaturalist asks for no more than 100 requests a minute
SEX_FEMALE = {"term_id": 9, "term_value_id": 10}

# iNaturalist serves photos whatever licence the photographer chose, and a null
# license_code means all rights reserved. Those must not be displayed, so the
# filter is applied in the request rather than after — otherwise the best-voted
# photo is usually the one we cannot use, and we would fall back to nothing.
# ND is included: showing a photo unaltered is not a derivative work.
USABLE = "cc0,cc-by,cc-by-sa,cc-by-nd,cc-by-nc,cc-by-nc-sa,cc-by-nc-nd"


def ebird_names(codes):
    """species_code -> scientific name, from the eBird taxonomy."""
    r = requests.get("https://api.ebird.org/v2/ref/taxonomy/ebird",
                     headers={"X-eBirdApiToken": os.environ["EBIRD_API_KEY"].strip()},
                     params={"fmt": "json", "species": ",".join(codes)}, timeout=60)
    r.raise_for_status()
    return {t["speciesCode"]: {"sci": t["sciName"], "com": t["comName"]}
            for t in r.json()}


def inat_taxon(scientific):
    """iNaturalist taxon id for a scientific name.

    eBird carries subspecies and forms — 'Setophaga petechia aestiva' — which
    iNaturalist may not have as its own taxon. Falling back to the binomial is
    right: the parent species is what a beginner is choosing between anyway.
    """
    for name in (scientific, " ".join(scientific.split()[:2])):
        r = requests.get(f"{INAT}/taxa", params={"q": name, "rank": "species",
                                                 "per_page": 1}, timeout=30)
        time.sleep(PAUSE)
        if r.ok and r.json().get("results"):
            return r.json()["results"][0]["id"]
    return None


def photo_for(taxon_id, **filters):
    """Best-voted research-grade observation photo matching the filters."""
    params = {"taxon_id": taxon_id, "photos": "true",
              "quality_grade": "research", "order_by": "votes",
              "photo_license": USABLE, "per_page": 1, **filters}
    r = requests.get(f"{INAT}/observations", params=params, timeout=30)
    time.sleep(PAUSE)
    if not r.ok:
        return None
    results = r.json().get("results") or []
    if not results or not results[0].get("photos"):
        return None
    p = results[0]["photos"][0]
    # belt and braces: the filter should make this impossible, but a photo with
    # no licence must never reach the app
    if not p.get("license_code"):
        return None
    return {
        # iNaturalist serves sizes by filename; square is the default in the
        # response and far too small for a grid
        "url": p["url"].replace("/square.", "/medium."),
        "licence": p.get("license_code"),
        "attribution": p.get("attribution"),
    }


if __name__ == "__main__":
    codes = sorted(json.loads(COEF.read_text()))
    cache = json.loads(OUT.read_text()) if OUT.exists() else {}
    names = ebird_names(codes)

    calls = 0
    for i, code in enumerate(codes, 1):
        entry = cache.setdefault(code, {})
        info = names.get(code)
        if not info:
            print(f"[{i}/{len(codes)}] {code}: not in the eBird taxonomy, skipped")
            continue
        entry["common_name"] = info["com"]
        entry["scientific_name"] = info["sci"]

        if not entry.get("taxon_id"):
            entry["taxon_id"] = inat_taxon(info["sci"])
            calls += 1
        if not entry["taxon_id"]:
            print(f"[{i}/{len(codes)}] {info['com']}: no iNaturalist match")
            continue

        months = entry.setdefault("months", {})
        missing = [m for m in range(1, 13) if str(m).zfill(2) not in months]
        for m in missing:
            got = photo_for(entry["taxon_id"], month=m)
            calls += 1
            if got:
                months[str(m).zfill(2)] = got

        if "female" not in entry:
            female = photo_for(entry["taxon_id"], **SEX_FEMALE)
            calls += 1
            if female:
                entry["female"] = female

        print(f"[{i}/{len(codes)}] {info['com']:32} "
              f"{len(months)}/12 months"
              f"{'  + female' if entry.get('female') else ''}", flush=True)
        OUT.write_text(json.dumps(cache, separators=(",", ":")))   # save as we go

    have = sum(1 for e in cache.values() if e.get("months"))
    print(f"\n{have} of {len(codes)} species have photos, {calls} requests made")
    print(f"wrote {OUT.name} ({OUT.stat().st_size / 1024:.0f} KB)")
