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

Split per species on purpose: the whole model is 207 KB gzipped, a single
species is under 4 KB, and the app only ever needs the one the user picked.
"""
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
BUNDLE = DATA / "public_bundle"

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
    }, indent=1))
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
    names = json.loads((DATA / "species_names.json").read_text())
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
