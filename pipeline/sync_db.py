"""Keep one checklist database shared between this laptop and CI.

    python pipeline/sync_db.py pull     remote rows -> local
    python pipeline/sync_db.py push     local file  -> remote

WHY THIS EXISTS
data/ is gitignored, because eBird records are not redistributable and the
repository is public. Without somewhere else to live, every machine starts from
an empty database and can only ever hold the rolling seven-day scrape window —
a runner would never accumulate the months of history the model needs, and the
laptop's history would never reach the runner. A private Hugging Face dataset
is the shared home: free, and outside git.

WHY MERGE RATHER THAN OVERWRITE
Both machines write. A runner scrapes daily; the laptop scrapes ad hoc and
refits. Downloading and replacing would silently destroy whichever side ran
last. Instead every table has a natural key, so rows missing locally are
inserted and rows already present are left alone — the same idempotence the
scraper already relies on when it re-reads overlapping days.

Without HF_DATASET and HF_TOKEN both commands do nothing and exit zero, so a
local-only checkout still runs the pipeline unchanged.
"""
import os
import shutil
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
DB = ROOT / "data" / "birding.duckdb"
REMOTE_NAME = "birding.duckdb"

# table -> the columns that identify a row uniquely
KEYS = {
    "checklists": ("sub_id",),
    "locations": ("loc_id",),
    "observations": ("sub_id", "species_code"),
}


def creds():
    repo = os.getenv("HF_DATASET")
    token = os.getenv("HF_TOKEN")
    if not (repo and token):
        return None, None
    return repo.strip(), token.strip()


def counts(path):
    if not Path(path).exists():
        return {}
    con = duckdb.connect(str(path), read_only=True)
    have = {r[0] for r in con.execute(
        "select table_name from information_schema.tables "
        "where table_schema = 'main'").fetchall()}
    out = {t: con.execute(f"select count(*) from {t}").fetchone()[0]
           for t in KEYS if t in have}
    con.close()
    return out


def merge(local, incoming):
    """Insert rows from `incoming` that `local` does not already have."""
    con = duckdb.connect(str(local))
    con.execute(f"attach '{incoming}' as incoming (read_only)")
    added = {}
    # an attached database is a CATALOG, not a schema — its tables carry
    # table_schema 'main' like everything else, so filtering on table_schema
    # matches nothing and every table gets skipped without complaint
    remote_tables = {r[0] for r in con.execute(
        "select table_name from information_schema.tables "
        "where table_catalog = 'incoming'").fetchall()}

    for tbl, keys in KEYS.items():
        if tbl not in remote_tables:
            continue
        # column order must agree for the positional insert below; both files
        # were created by the same code, so a mismatch means one of them is
        # from an older schema and merging would shift values between columns
        cols = [c[0] for c in con.execute(f"describe {tbl}").fetchall()]
        rcols = [c[0] for c in
                 con.execute(f"describe incoming.main.{tbl}").fetchall()]
        if cols != rcols:
            sys.exit(f"{tbl}: schema differs between local and remote.\n"
                     f"  local  {cols}\n  remote {rcols}\n"
                     "Re-scrape or migrate before syncing.")
        on = " and ".join(f"l.{k} = r.{k}" for k in keys)
        before = con.execute(f"select count(*) from {tbl}").fetchone()[0]
        con.execute(f"""
            insert into {tbl}
            select r.* from incoming.main.{tbl} r
            where not exists (select 1 from {tbl} l where {on})
        """)
        after = con.execute(f"select count(*) from {tbl}").fetchone()[0]
        added[tbl] = after - before
    con.close()
    return added


def pull(repo, token):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    try:
        got = hf_hub_download(repo_id=repo, filename=REMOTE_NAME,
                              repo_type="dataset", token=token)
    except (EntryNotFoundError, RepositoryNotFoundError):
        print(f"no {REMOTE_NAME} in {repo} yet — nothing to pull. "
              "The first push will create it.")
        return

    DB.parent.mkdir(exist_ok=True)
    if not DB.exists():
        shutil.copy(got, DB)
        print(f"no local database — took the remote one whole: {counts(DB)}")
        return

    # hf_hub_download hands back a path inside its cache; DuckDB would want to
    # write a WAL beside it, so attach a copy we own instead
    tmp = DB.parent / "_incoming.duckdb"
    shutil.copy(got, tmp)
    try:
        print(f"local  {counts(DB)}")
        print(f"remote {counts(tmp)}")
        added = merge(DB, tmp)
        print(f"merged in {added}")
    finally:
        tmp.unlink(missing_ok=True)


def push(repo, token):
    from huggingface_hub import HfApi
    if not DB.exists():
        sys.exit("no data/birding.duckdb to push — run the scrape first.")
    mb = DB.stat().st_size / 1024**2
    HfApi(token=token).upload_file(
        path_or_fileobj=str(DB), path_in_repo=REMOTE_NAME,
        repo_id=repo, repo_type="dataset",
        commit_message=f"sync: {counts(DB)}")
    print(f"pushed {REMOTE_NAME} ({mb:.1f} MB) to {repo} — {counts(DB)}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pull"
    repo, token = creds()
    if not repo:
        print("HF_DATASET / HF_TOKEN not set — skipping sync, staying local.")
        raise SystemExit(0)
    {"pull": pull, "push": push}[cmd](repo, token)
