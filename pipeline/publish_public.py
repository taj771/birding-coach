"""Publish the served model to a PUBLIC Hugging Face dataset.

    python pipeline/publish_public.py

WHY A SECOND, PUBLIC DATASET
The checklist database is private because eBird records are not
redistributable. Fitted coefficients are derived work and are fine to serve —
that distinction is what makes this whole app possible. Keeping them in the
private dataset would force the app to carry a Hugging Face token in the
browser, so they go somewhere with no token at all.

WHY NOT SUPABASE
Nothing is wrong with pushing to Supabase; it stays supported. But it makes
every refit a database migration, and a browser can fetch a static JSON file
from a CDN without one. The app then reads whatever the last monthly run
published, with no loading step between the two.

LAYOUT
    species.json            the picker: code -> common name
    hotspots.json           192 sites with coordinates and visit counts
    meta.json               when it was fitted, and on how much
    coef/<species>.json     one file per species, about 4 KB gzipped
    calibration.json        how the PREVIOUS model scored on data it never saw

Split per species on purpose: the whole model is 207 KB gzipped, a single
species is under 4 KB, and the app only ever needs the one the user picked.
"""
import json
import os
import re
import shutil
import sys

import requests
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
BUNDLE = DATA / "public_bundle"

# Beyond this from the nearest modelled hotspot, the app should decline rather
# than fall back to the pooled baseline. Twenty-five kilometres because sites
# sit roughly ten to twenty apart inside a covered county, so it reaches across
# a county that has been scraped without reaching into one that has not.
COVERAGE_RADIUS_KM = float(os.getenv("COVERAGE_RADIUS_KM", "25"))

SCALARS = ("intercept", "log_dur", "log_dist", "n_observers", "traveling",
           "wind_max", "temp_mean", "precip_sum", "cloud_mean")


def flat(m):
    """Nested model dict -> the flat term lookup serve/score.ts expects."""
    out = {k: float(v) for k, v in m.items() if k in SCALARS}
    for h, v in m.get("hour", {}).items():
        out[f"hour_{int(h):02d}"] = float(v)
    for w, v in m.get("week", {}).items():
        out[f"week_{int(w):02d}"] = float(v)
    for s, v in m.get("site", {}).items():
        out[f"site_{s}"] = float(v)
    for h, n in m.get("hour_support", {}).items():
        out[f"support_{int(h):02d}"] = float(n)
    meta = m.get("_meta", {})
    if "detections" in meta:
        out["meta_detections"] = float(meta["detections"])
    for split in ("by_date", "by_site"):
        for k, v in (meta.get(split) or {}).items():
            out[f"meta_{split}_{k}"] = float(v)
    return out


GAZ_URL = ("https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
           "2023_Gazetteer/2023_Gaz_zcta_national.zip")
GAZ_CACHE = DATA / "gaz_zcta.txt"
# How far outside the hotspots to keep postal codes. Someone just beyond the
# edge should be able to type their code and be told they are out of coverage,
# which is a useful answer. "Postal code not found" is not — it reads as a
# broken search box rather than a limit of the model.
ZIP_MARGIN_DEG = 0.6


def write_zips(hs):
    """Postal code centroids near the coverage, for the map's location box.

    Bundled rather than looked up at runtime. A geocoding service would work,
    but it puts a third party between a user typing their postal code and the
    map moving — one more thing to be rate limited, blocked by CORS, or simply
    down. This is 70 KB of static JSON served from the same place as the model,
    it works offline, and it cannot start charging.

    Source is the Census ZCTA gazetteer, public domain. Spot-checked against an
    independent geocoder: 15217 resolves to 40.4308, -79.9201 here and
    40.4308, -79.9205 there.

    Filtered to the bounding box of the fitted hotspots plus a margin, so this
    grows with the model instead of being pinned to a hand-typed list of
    prefixes. ZCTAs cross state lines and the file carries no state column, so
    geography is the only honest filter anyway.
    """
    import io
    import zipfile

    if not GAZ_CACHE.exists():
        print("  fetching the ZCTA gazetteer (once)...", flush=True)
        r = requests.get(GAZ_URL, timeout=180)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".txt"))
            GAZ_CACHE.write_bytes(z.read(name))

    lats = [h["latitude"] for h in hs]
    lons = [h["longitude"] for h in hs]
    lo_la, hi_la = min(lats) - ZIP_MARGIN_DEG, max(lats) + ZIP_MARGIN_DEG
    lo_lo, hi_lo = min(lons) - ZIP_MARGIN_DEG, max(lons) + ZIP_MARGIN_DEG

    out = {}
    with GAZ_CACHE.open() as f:
        header = [c.strip() for c in f.readline().split("\t")]
        gi, la_i, lo_i = (header.index("GEOID"), header.index("INTPTLAT"),
                          header.index("INTPTLONG"))
        for line in f:
            p = line.split("\t")
            try:
                la, lo = float(p[la_i]), float(p[lo_i])
            except ValueError:
                continue
            if lo_la <= la <= hi_la and lo_lo <= lo <= hi_lo:
                out[p[gi].strip()] = [round(la, 4), round(lo, 4)]

    (BUNDLE / "zips.json").write_text(json.dumps(out, separators=(",", ":")))
    print(f"  zips.json: {len(out):,} postal codes near the coverage "
          f"({(BUNDLE / 'zips.json').stat().st_size / 1024:.0f} KB)")


def species_names(codes):
    """species_code -> common name, derived rather than read from disk.

    This used to load data/species_names.json, which nothing in the pipeline
    ever wrote — it was a file made by hand on a laptop. data/ is gitignored,
    so a CI runner reached the last step of a ten-minute refit and died on a
    missing 2 KB file. The local copy was also stale: 69 names against a model
    that had grown to 89 species.

    fetch_photos.py already resolves a common name for every fitted species
    and runs earlier in the same job, so the names are on disk by the time we
    get here. The eBird taxonomy fills anything it missed, and the file is
    written back afterwards so the local preview tool still finds one.
    """
    names = {}

    photos = DATA / "photos.json"
    if photos.exists():
        for sp, e in json.loads(photos.read_text()).items():
            if e.get("common_name"):
                names[sp] = e["common_name"]

    missing = [c for c in codes if c not in names]
    if missing and os.getenv("EBIRD_API_KEY"):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import ebird_api
            taxa = ebird_api.get("/ref/taxonomy/ebird", fmt="json",
                                 species=",".join(missing)) or []
            for t in taxa:
                if t.get("speciesCode") and t.get("comName"):
                    names[t["speciesCode"]] = t["comName"]
        except Exception as e:
            print(f"  taxonomy lookup failed ({type(e).__name__}), "
                  f"falling back to species codes")

    still = [c for c in codes if c not in names]
    if still:
        # A code is an ugly label but a working one. Failing the publish over a
        # display string would be the wrong trade.
        print(f"  {len(still)} species have no common name, showing the code: "
              f"{', '.join(still[:6])}{' ...' if len(still) > 6 else ''}")
        names.update({c: c for c in still})

    (DATA / "species_names.json").write_text(
        json.dumps(names, indent=1, sort_keys=True))
    return names


def hotspots():
    """Parse the generated SQL rather than re-querying — one source of truth."""
    sql = (DATA / "hotspots.sql").read_text()
    rows = re.findall(r"\('(.+?)', '(.*?)', ([-\d.]+), ([-\d.]+), (\d+)\)", sql)
    return [{"loc_id": a, "name": b.replace("''", "'"),
             "latitude": float(c), "longitude": float(d), "n_checklists": int(e)}
            for a, b, c, d, e in rows]


def build(models, names):
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "coef").mkdir(parents=True)

    flattened = {sp: flat(m) for sp, m in models.items()}
    for sp, terms in flattened.items():
        (BUNDLE / "coef" / f"{sp}.json").write_text(
            json.dumps(terms, separators=(",", ":")))

    # Every species in one file, for the log form: ranking which birds are
    # likely at a spot means scoring all of them, and 69 separate requests to
    # do it once is worse than a single 200 KB one that the CDN then caches.
    # The forecast still fetches one species at a time — it only ever needs one.
    (BUNDLE / "all.json").write_text(json.dumps(flattened, separators=(",", ":")))

    # Photos, if they have been resolved. Kept in their own file rather than
    # folded into species.json: the forecast needs the species list on every
    # load and has no use for pictures, while the log form wants them only when
    # someone opens the identification helper.
    photos = {}
    src = DATA / "photos.json"
    if src.exists():
        raw = json.loads(src.read_text())
        # Only the curated photo is published. The grid shows one picture per
        # species; month and female variants are kept in data/photos.json so
        # they can be added back without refetching nine hundred images.
        photos = {sp: {"main": e["main"][:1]}
                  for sp, e in raw.items() if e.get("main")}
        (BUNDLE / "photos.json").write_text(json.dumps(photos, separators=(",", ":")))

    (BUNDLE / "species.json").write_text(json.dumps(
        sorted(({"species_code": sp,
                 "common_name": names.get(sp, sp),
                 "has_photos": sp in photos}
                for sp in models), key=lambda d: d["common_name"]),
        separators=(",", ":")))

    hs = hotspots()
    (BUNDLE / "hotspots.json").write_text(json.dumps(hs, separators=(",", ":")))
    write_zips(hs)

    weeks = sorted(int(w) for w in next(iter(models.values()))["week"])
    hours = sorted(int(h) for h in next(iter(models.values()))["hour"])
    (BUNDLE / "meta.json").write_text(json.dumps({
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "region": "US-PA-003",
        "region_name": "Allegheny County, Pennsylvania",
        "extent": {"latMin": 40.19, "latMax": 40.68,
                   "lonMin": -80.34, "lonMax": -79.69},
        "species": len(models),
        "hotspots": len(hs),
        "modelled_weeks": weeks,
        "modelled_hours": hours,
        # How far from a modelled hotspot a forecast still means anything.
        #
        # `extent` is a rectangle, and a rectangle stopped being an honest
        # description of coverage the moment this went past one county. With
        # Philadelphia in the east and Pittsburgh in the west, the bounding box
        # covers the whole state — including the northern tier, where there is
        # no data at all. A user there would pass the box test, match no
        # hotspot, fall back to the pooled baseline, and be shown a confident
        # number built from two cities four hundred kilometres apart.
        #
        # So the app should test distance to the nearest site in hotspots.json
        # instead, which describes a ragged coverage shape correctly. Every
        # coordinate needed for that is already published; only the threshold
        # was missing.
        #
        # extent stays, for framing the map. It must not be used to decide
        # whether a point can be forecast.
        "coverage_radius_km": COVERAGE_RADIUS_KM,
    }, indent=1))

    # The report card on the model this one is replacing, carried forward so
    # the claim and its check travel together. Published rather than left in a
    # run log because a calibration number nobody can see is a number nobody
    # acts on — and because a user is entitled to ask how often "40%" turns out
    # to mean 40%.
    #
    # Deliberately the PREVIOUS model's result: eval_calibration.py runs before
    # the refit, since the only honest hold-out is data the fitted model never
    # saw. So calibration.json always describes the model published last month,
    # not the one in the same folder. The `model_fitted_at` field inside it says
    # which, and it will not match meta.json's `fitted_at`.
    cal = DATA / "calibration.json"
    if cal.exists():
        (BUNDLE / "calibration.json").write_text(cal.read_text())
        print(f"  calibration.json carried forward "
              f"({json.loads(cal.read_text()).get('verdict', '?')})")
    else:
        print("  no calibration.json — eval has not run yet, or had too "
              "little held-out data")

    return hs, weeks


if __name__ == "__main__":
    repo = os.getenv("HF_PUBLIC_DATASET")
    token = os.getenv("HF_TOKEN")
    if not repo:
        raise SystemExit(
            "HF_PUBLIC_DATASET is not set.\n"
            "Create a PUBLIC dataset on Hugging Face and put its name in .env,\n"
            "e.g. HF_PUBLIC_DATASET=taj771/birding-coach-model\n"
            "It must be public: the app fetches from it with no token, which is\n"
            "the point. Only fitted coefficients go here, never eBird records.")

    models = json.loads((DATA / "model_coefficients.json").read_text())
    names = species_names(sorted(models))
    hs, weeks = build(models, names)

    size = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file())
    print(f"built bundle: {len(models)} species, {len(hs)} hotspots, "
          f"{len(weeks)} fitted weeks, {size/1024:.0f} KB total")

    if not token:
        raise SystemExit(f"HF_TOKEN not set — bundle left in {BUNDLE}")

    from huggingface_hub import HfApi
    api = HfApi(token=token.strip())
    info = api.repo_info(repo, repo_type="dataset")
    if info.private:
        raise SystemExit(
            f"{repo} is private. The app fetches these files without a token,\n"
            "so a private repo would return 401 to every visitor. Make it\n"
            "public in the dataset settings, or point HF_PUBLIC_DATASET at a\n"
            "public one.")

    api.upload_folder(folder_path=str(BUNDLE), repo_id=repo,
                      repo_type="dataset",
                      commit_message=f"model: {len(models)} species")
    base = f"https://huggingface.co/datasets/{repo}/resolve/main"
    print(f"pushed to {repo}")
    print(f"  {base}/species.json")
    print(f"  {base}/hotspots.json")
    print(f"  {base}/coef/norpar.json")
