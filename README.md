# Birding Coach

Every birding app tells you *where*. This one tells you *when*.

Pick a species and a place, and it returns the hours over the coming week when
you are most likely to actually see it — the weather forecast applied to what
birders have historically reported at those spots.

Study region is **Allegheny County, Pennsylvania**.

**New here?** [OVERVIEW.md](OVERVIEW.md) explains the whole system — the three
repositories, how a checklist becomes a number on a phone, and why it is built
this way. This README goes deeper on the model alone.

## Why the gap exists

Traveling Birder, BirdTrip, BirdPlan and eBird's own Hotspot Explorer all take
a target species and hand back a *place*. None of them says *go at 6:30, not
10:00*, or *budget ninety minutes, not thirty*.

Those two turn out to be the largest measurable effects in the data.

## What the data says

4,207 complete checklists, 48 days, 1,039 sites, 69 species modelled.

| | |
|---|---|
| Time spent birding | the dominant effect. Every species, large. |
| Time of day | real, and strongly species-dependent |
| Wind | negative wherever it is detectable |
| Rain, cloud | weak; significant for a minority of species |

Time-of-day matters far more for birds found by ear than for total species
count — most birds are not singing at dawn. Generic *"go early"* advice
underdelivers; species-specific timing is where the value is.

The model is best at exactly the birds people want help with. Warblers, which
are seasonal and detected by song, predict well — Nashville 0.95, Northern
Parula 0.93, Tennessee 0.93. Year-round residents predict worst: Song Sparrow
0.68, Chimney Swift 0.66. There is little to predict when a bird is everywhere
all the time.

## How it works

A penalised (ridge) logistic regression per species:

```
P(reported | hour, week, site, effort, weather)
```

Fitted only on **complete** checklists — lists where the birder declared they
recorded every species they could identify. That is what makes a probability
computable at all: a species missing from a complete list is a real absence,
not silence. Without it you have a tally of sightings and no denominator.

Ridge rather than plain maximum likelihood so that every one of the 192
hotspots can carry its own effect. A hotspot visited 89 times keeps its own
estimate; one visited four times is pulled toward the county average. Shrinkage
follows the evidence, which is what an arbitrary visit threshold was crudely
approximating — and unpenalised fitting separated badly, losing roughly half
the species to infinite coefficients once site effects were added.

The cost is inference: ridge estimates are biased on purpose, so there are no
valid p-values. Fine here, since the app needs point predictions. Claims about
whether an effect is *real* need the unpenalised fit, which is a separate job.

### Two evaluations, because there are two questions

```
new site   AUC 0.72, calibration MAE 0.038    somewhere never visited
new day    AUC 0.79, calibration MAE 0.030    a known hotspot, future date
```

Held out by site and by date respectively. The second is what the app actually
does, and it is the better of the two — which is the evidence that site effects
carry real information. Holding out only by site would have hidden that
entirely, since a site the model has never seen cannot benefit from site
effects by construction.

## Pipeline

```
python pipeline/run.py daily      scrape recent checklists       ~5 min
python pipeline/run.py monthly    refit and publish              ~10 min
```

Split by how fast things change. New checklists arrive continuously. How
detection responds to hour, effort and weather does not move week to week.

```
sync_db.py             pull and push the shared checklist database
scrape_ebird.py        two API calls per checklist — eBird has no bulk endpoint
backfill_locations.py  site coordinates, from the feed rather than per-site calls
build_model_table.py   weather joined per grid cell, over the hours each outing spans
fit_logit.py           one ridge logit per species -> model_coefficients.json
publish.py             -> Supabase, a HF dataset, or a SQL file to paste
```

### Where the database lives

Not in git. eBird records are not redistributable and this repository is
public, so `data/` is ignored — which leaves a scheduled runner with nowhere to
accumulate. Each run would scrape its rolling seven-day window into an empty
file and refit on one week of data.

The database lives in a **private Hugging Face dataset** instead. Every run
pulls it, appends what it scraped, and pushes it back. Both sides write, so the
pull merges rather than overwrites: each table has a natural key
(`sub_id`, `loc_id`, `(sub_id, species_code)`) and only missing rows are
inserted. Running it twice adds nothing, the same idempotence the scraper
already relies on when it re-reads overlapping days.

Set `HF_DATASET` and `HF_TOKEN` (a **write** token) and it happens
automatically. Leave them unset and the sync is skipped, so a local-only
checkout still works.

### Reproducibility

The same data was fitted on a laptop and on a GitHub runner and the two
coefficient sets compared term by term:

```
69 species, 17,526 terms, identical key sets
max absolute difference   7.9e-08
all terms agree within    1e-06
held-out metrics differ by at most 1.1e-09
```

Across arm64 and x86_64. The residual is lbfgs stopping a hair earlier or
later because a different BLAS sums the same numbers in a different order.

That check only held because both machines happened to install the same
library versions, so `requirements.txt` is pinned — including numpy and scipy,
which nothing imports directly but which do the arithmetic. A changed solver
default would otherwise move the published coefficients with nothing to
compare against. To upgrade: bump the pins, refit, and diff against the
previous run before publishing.

Scraping targets dates about a week old. eBird submissions trail the outing —
people enter a weekend trip on Monday — so scraping fresh gets only the
log-in-the-field crowd, which is a biased sample.

## Serving

The fitted model is a lookup table plus a sum, so the app needs no Python.
[`serve/score.ts`](serve/score.ts) is the whole scoring layer: look up the
hour, the week and the site, add the effort and weather terms, apply the
logistic transform.

The seven-day window rolls on its own — the app fetches its own forecast at
request time, so nothing needs to be pushed daily.

## Honest limits

- **Detection, not presence.** This predicts whether you would *report* a bird,
  not whether one is there. For a birder that is the right target. It is not
  the same quantity.
- **One county**, because the eBird API cannot supply more. Building a
  denominator costs two calls per checklist: ~4,650 for a month of one county,
  ~438,000 for a year of Pennsylvania, over a million for a month of the United
  States. Wider coverage means requesting the eBird Basic Dataset instead of
  scraping.
- **Effort is endogenous.** Birders extend a good outing, so duration is partly
  a consequence of success rather than only a cause. That inflates the effort
  coefficient.
- **Weather is measured on a 25 km grid**, which attenuates its coefficients
  toward zero. The wind effect is probably understated.
- **Observers self-select on weather.** Nobody birds in a storm, so the
  observed weather range is truncated and whoever does go out is unusual.
- **The feed truncates** any county-day above 200 checklists, under-sampling
  the busiest mornings.

## Data

eBird API 2.0, Cornell Lab of Ornithology — free for non-commercial use, must
be cited, and **not redistributable in raw form**. This repository contains no
eBird records; `data/` is gitignored. Serving model output is derived work and
is fine; serving raw checklists would not be.

Weather is ERA5 reanalysis and forecasts via [Open-Meteo](https://open-meteo.com).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add a key from ebird.org/api/keygen
.venv/bin/python pipeline/run.py all
```
