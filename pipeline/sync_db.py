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
        # Columns are named rather than taken positionally, because the two
        # files legitimately disagree on order: a column added later by ALTER
        # TABLE lands at the end, while a database built from scratch gets it
        # wherever CREATE TABLE declares it. Same schema, different layout —
        # and a positional insert would silently write each column's values
        # into its neighbour.
        cols = [c[0] for c in con.execute(f"describe {tbl}").fetchall()]
        rcols = [c[0] for c in
                 con.execute(f"describe incoming.main.{tbl}").fetchall()]
        if set(cols) != set(rcols):
            only_l = sorted(set(cols) - set(rcols))
            only_r = sorted(set(rcols) - set(cols))
            sys.exit(f"{tbl}: the two databases hold different columns.\n"
                     f"  only local  {only_l}\n  only remote {only_r}\n"
                     "One side is from an older schema. Migrate before syncing.")
        on = " and ".join(f"l.{k} = r.{k}" for k in keys)
        before = con.execute(f"select count(*) from {tbl}").fetchone()[0]
        con.execute(f"""
            insert into {tbl} ({", ".join(cols)})
            select {", ".join(f"r.{c}" for c in cols)}
            from incoming.main.{tbl} r
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


def remote_counts(api, repo):
    """What the last push said it was publishing, from the commit message.

    Reading the counts off the history rather than downloading the file: the
    database is twenty megabytes and a backfill pushes every fifteen
    region-days, so fetching it back each time to compare would cost more than
    the scrape. Best effort by design — an unreadable or unrecognised message
    returns None and the caller carries on rather than blocking a push.
    """
    import ast
    try:
        msg = api.list_repo_commits(repo, repo_type="dataset")[0].title
        return ast.literal_eval(msg.split("sync: ", 1)[1])
    except Exception:
        return None


def push(repo, token):
    from huggingface_hub import HfApi
    if not DB.exists():
        sys.exit("no data/birding.duckdb to push — run the scrape first.")

    # An upload replaces the file whole, so a local database smaller than the
    # published one is a destructive push about to happen — a runner that
    # scraped without pulling first, most likely. Every legitimate push grows
    # the table, because merge() only ever inserts and the scrape only ever
    # adds. Refuse rather than publish the loss and find out later.
    api = HfApi(token=token)
    there, here = remote_counts(api, repo), counts(DB)
    if there:
        shrunk = {t: (there[t], here.get(t, 0)) for t in there
                  if here.get(t, 0) < there[t]}
        if shrunk:
            sys.exit(
                f"refusing to push: this would shrink {repo}.\n" +
                "".join(f"  {t}: {r:,} published -> {l:,} local\n"
                        for t, (r, l) in shrunk.items()) +
                "The local database is missing rows the remote has, which "
                "means it was not pulled before scraping. Run\n"
                "  python pipeline/sync_db.py pull\n"
                "and push again. Nothing has been changed on the remote.")

    mb = DB.stat().st_size / 1024**2
    api.upload_file(
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
