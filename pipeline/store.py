"""The checklist store: partitioned Parquet on Hugging Face.

    python pipeline/store.py migrate     one-off, from birding.duckdb
    python pipeline/store.py status      what is stored, by region

WHY THIS REPLACES sync_db.py
sync_db pulled one DuckDB file at the start of every run and pushed the whole
thing back at the end. At 8,000 checklists that file is 13 MB and the round
trip is a few seconds. Pennsylvania at hotspot scale is about 213,000
checklists a year, which is roughly 355 MB — and the daily run would move all
of it, both ways, to add forty new rows.

Worse quietly: the GitHub Actions cache is capped at 10 GB per repository and
each run saves a new entry, so a 355 MB database evicts older entries after
about twenty-eight runs. An evicted cache does not error. The runner simply
starts with an empty data/ directory, scrapes into nothing, and reports
success — the exact failure sync_db was written to prevent.

LAYOUT
    index/days.parquet                          one row per region-day held
    checklists/region=US-PA-003/2025-08.parquet
    observations/region=US-PA-003/2025-08.parquet
    locations/region=US-PA-003.parquet

Partitioned by region and month because that is how the work is shaped: a
scrape touches one county and a handful of days, so it rewrites one or two
small files rather than the whole database. Two counties scraped at once write
different files and cannot clobber each other, which is what makes parallel
county jobs possible at all.

READING COSTS NOTHING TO SET UP
DuckDB reads Hugging Face directly, so the monthly fit does not download
anything first:

    SELECT * FROM read_parquet(
      'hf://datasets/<ds>/checklists/**/*.parquet', hive_partitioning = true)

Hive partitioning means `region` comes back as a column, and a query filtered
to one county reads only that county's files.

THE DAYS INDEX
Resuming a backfill means knowing which region-days are already held. Deriving
that from the partitions themselves would mean opening every file, so it is
maintained alongside as one small Parquet. It is rebuilt from the partitions by
`migrate` and by `status --rebuild`, so a drifted index is repairable rather
than authoritative.
"""
import os
import sys
from pathlib import Path

import duckdb
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

DATASET = (os.getenv("HF_DATASET") or "").strip()
TOKEN = (os.getenv("HF_TOKEN") or "").strip()
LOCAL_DB = ROOT / "data" / "birding.duckdb"

TABLES = ("checklists", "observations", "locations")
# locations are per region, not per month — a site does not belong to a month
MONTHLY = ("checklists", "observations")


def base():
    if not DATASET:
        sys.exit("HF_DATASET is not set — the checklist store lives there.")
    return f"hf://datasets/{DATASET}"


def connect(local=None):
    """A DuckDB connection wired up to read the store."""
    con = duckdb.connect(str(local) if local else ":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    if TOKEN:
        # Escaped rather than interpolated raw: a token is opaque and could in
        # principle contain a quote.
        con.execute("CREATE OR REPLACE SECRET hf "
                    f"(TYPE HUGGINGFACE, TOKEN '{TOKEN.replace(chr(39), chr(39) * 2)}')")
    return con


def glob_for(table):
    if table in MONTHLY:
        return f"{base()}/{table}/region=*/*.parquet"
    return f"{base()}/{table}/region=*.parquet"


def views(con):
    """Expose every table as a view over its partitions.

    Missing tables become empty views rather than errors, so a fresh store and
    a full one behave the same to callers.
    """
    for t in TABLES:
        try:
            con.execute(f"""
                CREATE OR REPLACE VIEW {t} AS
                SELECT * FROM read_parquet('{glob_for(t)}',
                                           hive_partitioning = true,
                                           union_by_name = true)""")
        except duckdb.Error:
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT NULL WHERE false")
    return con


def days_held(con=None):
    """{(region, 'YYYY-MM-DD')} already scraped — the resume set."""
    own = con is None
    con = con or connect()
    try:
        rows = con.execute(
            f"SELECT region, obs_date FROM read_parquet('{base()}/index/days.parquet')"
        ).fetchall()
    except duckdb.Error:
        rows = []
    finally:
        if own:
            con.close()
    return {(r[0], str(r[1])) for r in rows}


def _upload(paths_and_targets, message):
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=TOKEN)
    ops = [CommitOperationAdd(path_in_repo=t, path_or_fileobj=str(p))
           for p, t in paths_and_targets]
    # One commit for the whole batch. Uploading file by file would make a
    # backfill chunk into fifteen separate dataset revisions.
    api.create_commit(repo_id=DATASET, repo_type="dataset",
                      operations=ops, commit_message=message)
    return len(ops)


def write(con, staging, regions=None, message="update"):
    """Export touched partitions from a local DuckDB and upload them.

    `con` must hold real tables (not the remote views) — this is the write
    side, used after a scrape has inserted into a local database.
    """
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    jobs = []

    # One partition list, derived from the checklists, and observations follow
    # it. Deriving them separately risks the two drifting apart — an
    # observation whose checklist landed in a different month's file would be
    # unreachable by any join that prunes on partition.
    parts = con.execute("""
        SELECT DISTINCT region, strftime(obs_date, '%Y-%m') AS month
        FROM checklists WHERE region IS NOT NULL AND obs_date IS NOT NULL
        ORDER BY 1, 2""").fetchall()
    if regions:
        parts = [p for p in parts if p[0] in set(regions)]

    for region, month in parts:
        # `region` is excluded from the payload because the path carries it —
        # hive_partitioning reads it back from region=<code>/ on the way in.
        # Writing it twice would make the column ambiguous on read.
        ck = staging / f"checklists-{region}-{month}.parquet"
        con.execute(f"""
            COPY (SELECT * EXCLUDE (region) FROM checklists
                  WHERE region = '{region}'
                    AND strftime(obs_date, '%Y-%m') = '{month}')
            TO '{ck}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        jobs.append((ck, f"checklists/region={region}/{month}.parquet"))

        ob = staging / f"observations-{region}-{month}.parquet"
        con.execute(f"""
            COPY (SELECT o.* FROM observations o
                  JOIN checklists c USING (sub_id)
                  WHERE c.region = '{region}'
                    AND strftime(c.obs_date, '%Y-%m') = '{month}')
            TO '{ob}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        jobs.append((ob, f"observations/region={region}/{month}.parquet"))

    # locations carry no date, so one file per region rather than per month.
    # A location is assigned to a region through the checklists filed there.
    for region in sorted({p[0] for p in parts}):
        out = staging / f"locations-{region}.parquet"
        con.execute(f"""
            COPY (SELECT DISTINCT l.* FROM locations l
                  WHERE l.loc_id IN (SELECT loc_id FROM checklists
                                     WHERE region = '{region}'))
            TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        jobs.append((out, f"locations/region={region}.parquet"))

    # Sites no checklist points at cannot be assigned a region, so they land
    # in their own partition instead of being dropped. There are only a handful
    # — backfill_locations resolves coordinates from a whole day's feed, which
    # covers locations whose checklists we did not end up keeping — but a store
    # that silently loses rows is not one you can verify a round trip against,
    # and unverifiable is how the column-order bug survived three scrapes.
    orphans = con.execute("""
        SELECT count(*) FROM locations l
        WHERE NOT EXISTS (SELECT 1 FROM checklists c WHERE c.loc_id = l.loc_id)
    """).fetchone()[0]
    if orphans and not regions:
        out = staging / "locations-_unassigned.parquet"
        con.execute(f"""
            COPY (SELECT DISTINCT l.* FROM locations l
                  WHERE NOT EXISTS (SELECT 1 FROM checklists c
                                    WHERE c.loc_id = l.loc_id))
            TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        jobs.append((out, "locations/region=_unassigned.parquet"))
        print(f"  {orphans} location(s) referenced by no checklist -> _unassigned")

    idx = staging / "days.parquet"
    con.execute(f"""
        COPY (SELECT region, obs_date, count(*) AS n_checklists, now() AS updated_at
              FROM checklists GROUP BY 1, 2 ORDER BY 1, 2)
        TO '{idx}' (FORMAT PARQUET)""")
    jobs.append((idx, "index/days.parquet"))

    n = _upload(jobs, message)
    print(f"uploaded {n} file(s) to {DATASET}")
    return n


def migrate():
    """One-off: existing birding.duckdb -> partitioned Parquet.

    Reads the database wherever it is. The canonical copy lives in the Hugging
    Face dataset and DuckDB can attach that directly over https, so this does
    not need a local checkout to have pulled it first.
    """
    con = connect()
    src = f"{base()}/birding.duckdb"
    if LOCAL_DB.exists() and LOCAL_DB.stat().st_size > 0:
        src = str(LOCAL_DB)
    print(f"reading {src}")
    con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
    for t in TABLES:
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM src.{t}")

    n_ck, n_ob = con.execute(
        "SELECT (SELECT count(*) FROM checklists), "
        "       (SELECT count(*) FROM observations)").fetchone()
    print(f"  {n_ck:,} checklists, {n_ob:,} observations")

    staging = ROOT / "data" / "_staging"
    write(con, staging, message=f"migrate to partitioned parquet: {n_ck:,} checklists")
    con.close()


def status(rebuild=False):
    con = views(connect())
    try:
        rows = con.execute("""
            SELECT region, count(*) AS checklists,
                   count(DISTINCT obs_date) AS days,
                   min(obs_date) AS first, max(obs_date) AS last
            FROM checklists GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    except duckdb.Error as e:
        sys.exit(f"nothing readable in the store yet ({type(e).__name__})")
    print(f"{'region':12} {'checklists':>11} {'days':>6}  span")
    for r in rows:
        print(f"{r[0]:12} {r[1]:>11,} {r[2]:>6}  {r[3]} .. {r[4]}")
    total = sum(r[1] for r in rows)
    print(f"{'TOTAL':12} {total:>11,}")
    con.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "migrate":
        migrate()
    elif cmd == "status":
        status()
    else:
        sys.exit(__doc__.split("\n\n")[0] + "\n\nusage: store.py [migrate|status]")
