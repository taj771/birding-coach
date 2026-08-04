/**
 * Scoring for Birding Coach — hand this to Lovable.
 *
 * The fitted model is a lookup table plus a sum, so nothing here needs a
 * machine-learning runtime. Coefficients come from the `model_coefficients`
 * table, written monthly by the Python pipeline; the weather comes from
 * Open-Meteo at request time, which is why the seven-day window advances on
 * its own without anything being pushed daily.
 *
 * Term names must match pipeline/publish.py exactly:
 *   intercept, log_dur, log_dist, n_observers, traveling,
 *   wind_max, temp_mean, precip_sum, cloud_mean,
 *   hour_06 .. hour_20, week_00 .. week_52,
 *   site_<locId> for every hotspot, site_PERSONAL as the fallback,
 *   support_06 .. support_20, meta_detections,
 *   meta_by_date_auc / _cal_mae   (known hotspot, future day — the app's job)
 *   meta_by_site_auc / _cal_mae   (a location the model has never seen)
 */

type Coef = Record<string, number>;

interface Outing {
  durationHrs: number;
  distanceKm: number;
  observers: number;
  traveling: boolean;
}

interface HourWeather {
  windMax: number;      // max over the hours the outing spans
  tempMean: number;
  precipSum: number;
  cloudMean: number;
}

/** Week index must match Python's `doy // 7`. */
export function weekOf(d: Date): number {
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const doy = Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(),
                                   d.getUTCDate()) - start) / 86_400_000);
  return Math.floor(doy / 7);
}

const pad = (n: number) => String(n).padStart(2, "0");

/**
 * Why a cell cannot be scored, or null when it can.
 *
 * The week check is not defensive padding. Scraping covers 48 days spread
 * across 18 months, so most calendar weeks have no fitted term at all. Reading
 * a missing week as 0 does not mean "average week" — it silently removes
 * seasonality from the sum, and seasonality is most of what the model knows
 * about migrants. A Northern Parula would be scored in August using everything
 * except the fact that it is August.
 */
export function unavailable(
  c: Coef, date: Date, hour: number, minSupport = 50,
): "hour" | "week" | null {
  if ((c[`support_${pad(hour)}`] ?? 0) < minSupport) return "hour";
  if (c[`week_${pad(weekOf(date))}`] === undefined) return "week";
  return null;
}

/** Which weeks the model can speak to at all, for the UI to show up front. */
export function modelledWeeks(c: Coef): number[] {
  return Object.keys(c)
    .filter(k => k.startsWith("week_"))
    .map(k => Number(k.slice(5)))
    .sort((a, b) => a - b);
}

/**
 * One cell of the grid. Returns null when the model cannot back a number —
 * too little data for that hour, or no fitted term for that week. The UI greys
 * those; see `unavailable` for which of the two it was.
 */
export function score(
  c: Coef, date: Date, hour: number, o: Outing, w: HourWeather,
  locId?: string, minSupport = 50,
): number | null {
  if (unavailable(c, date, hour, minSupport)) return null;

  // a known hotspot uses its own effect; anywhere else falls back to the
  // pooled baseline, and the UI should say the number is county-wide
  const site = (locId && c[`site_${locId}`] !== undefined)
    ? c[`site_${locId}`]
    : (c.site_PERSONAL ?? 0);

  const logit =
    (c.intercept ?? 0) +
    (c[`hour_${pad(hour)}`] ?? 0) +
    (c[`week_${pad(weekOf(date))}`] ?? 0) +
    site +
    (c.log_dur ?? 0) * Math.log(o.durationHrs) +
    (c.log_dist ?? 0) * Math.log1p(o.distanceKm) +
    (c.n_observers ?? 0) * o.observers +
    (c.traveling ?? 0) * (o.traveling ? 1 : 0) +
    (c.wind_max ?? 0) * w.windMax +
    (c.temp_mean ?? 0) * w.tempMean +
    (c.precip_sum ?? 0) * w.precipSum +
    (c.cloud_mean ?? 0) * w.cloudMean;

  return 1 / (1 + Math.exp(-logit));   // the only real maths in the file
}

/**
 * What each choice was worth, in percentage points, against a reference
 * outing (10:00, 30 minutes, median wind). Computed by re-scoring with one
 * term reset, NOT by reading coefficients — those are log-odds and would not
 * sum to the number on screen.
 */
export function drivers(
  c: Coef, date: Date, hour: number, o: Outing, w: HourWeather,
  locId?: string,
) {
  // minSupport 0 here on purpose: the caller has already decided this cell
  // is worth showing, and the breakdown must not disagree with the number
  const at = (h: number, out: Outing, wx: HourWeather) =>
    score(c, date, h, out, wx, locId, 0)!;
  const actual = at(hour, o, w);

  const refHour = 10, refDur = 0.5, refWind = 12;
  const base = at(refHour, { ...o, durationHrs: refDur },
                  { ...w, windMax: refWind });

  return [
    { label: "time of day", pts: at(hour, { ...o, durationHrs: refDur },
                                    { ...w, windMax: refWind }) - base },
    { label: "how long",    pts: at(refHour, o,
                                    { ...w, windMax: refWind }) - base },
    { label: "weather",     pts: at(refHour, { ...o, durationHrs: refDur },
                                    w) - base },
  ].map(d => ({ ...d, pts: Math.round(d.pts * 100) }))
   .concat([{ label: "total", pts: Math.round(actual * 100) }]);
}

/** The full grid: 7 days x the hours the model covers. */
export function grid(
  c: Coef, forecast: { time: string; wind: number; temp: number;
                       precip: number; cloud: number }[],
  o: Outing, locId?: string,
  hours = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
) {
  const byKey = new Map(forecast.map(f => [f.time.slice(0, 13), f]));
  const days = [...new Set(forecast.map(f => f.time.slice(0, 10)))].slice(0, 7);

  return days.flatMap(day =>
    hours.map(h => {
      // the outing spans several hours; wind is the MAX over them, because a
      // gust that masks birdsong matters more than the average
      const span = Array.from({ length: Math.max(1, Math.ceil(o.durationHrs)) },
                              (_, i) => byKey.get(`${day}T${pad(h + i)}`))
                        .filter(Boolean) as typeof forecast;
      if (!span.length) {
        return { day, hour: h, probability: null, reason: "forecast" as const };
      }

      const w: HourWeather = {
        windMax: Math.max(...span.map(s => s.wind)),
        tempMean: span.reduce((a, s) => a + s.temp, 0) / span.length,
        precipSum: span.reduce((a, s) => a + s.precip, 0),
        cloudMean: span.reduce((a, s) => a + s.cloud, 0) / span.length,
      };
      const date = new Date(day);
      return { day, hour: h,
               probability: score(c, date, h, o, w, locId),
               reason: unavailable(c, date, h) };
    }));
}

/** True when the model knows this specific place, rather than falling back
 *  to the county baseline. The UI should mark the difference — a pin on an
 *  unmodelled spot is an Allegheny-wide estimate, not a local one. */
export function hasSiteEffect(c: Coef, locId: string): boolean {
  return c[`site_${locId}`] !== undefined;
}

/** Bounding box of the sites the model was fitted on: Allegheny County, PA,
 *  about 53 km across either way. */
export const EXTENT = { latMin: 40.19, latMax: 40.68,
                        lonMin: -80.34, lonMax: -79.69 };

/**
 * Whether a point is somewhere this model can speak about at all.
 *
 * Nothing in `score` knows where it is. A pin in Philadelphia matches none of
 * the 192 hotspot terms, falls back to the pooled `site_PERSONAL` baseline,
 * and returns a confident number built entirely from Allegheny County's
 * intercept, hours and weeks. The failure is invisible unless it is checked
 * here, so check it before scoring rather than trusting the caller.
 */
export function inCoverage(lat: number, lon: number): boolean {
  return lat >= EXTENT.latMin && lat <= EXTENT.latMax
      && lon >= EXTENT.lonMin && lon <= EXTENT.lonMax;
}
