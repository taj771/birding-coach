"""One eBird HTTP client, shared by every script that calls the API.

WHY THIS EXISTS
The retry logic used to live in scrape_ebird.py only. backfill_locations.py
called requests directly with raise_for_status(), so a single 429 killed the
whole run — and because backfill.py runs the two in sequence, that threw away
whatever the chunk had scraped but not yet pushed.

Two copies of "how do we talk to eBird" is how one of them silently stops
matching the other. There is one now.

RATE LIMITS
eBird publishes no numeric limit, and 429 carries no documented Retry-After,
so the backoff is deliberately generous: a minute, then two, then four. Waiting
four minutes is cheap next to losing a chunk of scraping and starting the day
again. Retry-After is honoured when the server does send it.

eBird's terms prohibit use that "adversely impacts the stability of the
ebird.org servers", and the route they intend for bulk pulls is the EBD
download rather than this API. Every call site should stay slow on purpose.
"""
import os
import sys
import time

import requests

BASE = "https://api.ebird.org/v2"

DELAY = float(os.getenv("SCRAPE_DELAY", "0.5"))

# Escalating, not fixed: a rate limit that a 30 s pause does not clear is not
# cleared by another 30 s pause either. Doubling gets out of the way of a
# genuinely busy server instead of drumming on it.
BACKOFF = (60, 120, 240)

_sess = None


def session():
    """The shared session, built on first use.

    Deliberately lazy. Reading the key at import time would make this module
    depend on being imported *after* the caller's load_dotenv(), which is the
    kind of ordering that works until somebody moves an import line.
    """
    global _sess
    if _sess is None:
        # .strip() matters: a secret pasted with a trailing newline makes
        # requests raise InvalidHeader on every call, which looks exactly like
        # an API outage rather than a bad secret
        key = (os.getenv("EBIRD_API_KEY") or "").strip()
        if not key:
            sys.exit("EBIRD_API_KEY not set — put it in .env locally, or in\n"
                     "GitHub Settings > Secrets and variables > Actions")
        _sess = requests.Session()
        _sess.headers.update({"X-eBirdApiToken": key})
    return _sess


def _wait_after(resp, fallback):
    """Honour Retry-After when present, otherwise use our own backoff."""
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return max(float(raw), 1.0)
        except ValueError:
            pass          # it may be an HTTP date; not worth parsing
    return fallback


def get(path, **params):
    """One GET, with retries. Returns parsed JSON, or None if it never
    succeeded.

    Returning None rather than raising is deliberate. A missing day should
    leave a gap the caller can report and carry on from — the alternative is
    that one bad response ends a run that has hours of good work behind it.
    Callers must treat None as "no data", never as "empty".
    """
    for attempt in range(len(BACKOFF) + 1):
        try:
            r = session().get(f"{BASE}{path}", params=params, timeout=45)

            if r.status_code == 200:
                time.sleep(DELAY)
                return r.json()

            # Every 4xx except the rate limit is terminal: the request itself
            # is the problem, so repeating it unchanged cannot help. eBird
            # answers a deleted or merged checklist with 410 rather than 404,
            # which is why this is a range and not a pair of literals — the
            # first version retried 410 four times before giving up.
            if 400 <= r.status_code < 500 and r.status_code != 429:
                if r.status_code not in (404, 410):
                    print(f"    HTTP {r.status_code} on {path}", flush=True)
                return None

            # 429 is the rate limit; 5xx is eBird having a bad minute. Both are
            # worth waiting out, and neither is worth failing the job over.
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == len(BACKOFF):
                    print(f"    HTTP {r.status_code} on {path} — "
                          f"giving up after {len(BACKOFF)} retries", flush=True)
                    return None
                pause = _wait_after(r, BACKOFF[attempt])
                print(f"    HTTP {r.status_code} on {path} — "
                      f"waiting {pause:.0f}s then retrying", flush=True)
                time.sleep(pause)
                continue

            print(f"    HTTP {r.status_code} on {path}", flush=True)

        except requests.RequestException as e:
            print(f"    {type(e).__name__} on {path}", flush=True)

        if attempt < len(BACKOFF):
            time.sleep(3 * (attempt + 1))

    return None


def day_feed(region, date):
    """Checklists filed in a region on one day.

    The 200-result cap is the API's, not ours; a busier day is silently
    truncated, which is why the caller warns when it sees exactly 200.
    """
    y, m, d = str(date).split("-")
    return get(f"/product/lists/{region}/{y}/{m}/{d}", maxResults=200)
