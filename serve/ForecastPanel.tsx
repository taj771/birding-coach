/**
 * The forecast panel, as built and checked locally against the real model.
 *
 * This is a transcription of data/preview.html — same layout, same ordering,
 * same greying rules — with one change: the weather sliders are gone and the
 * values come from Open-Meteo's hourly forecast for the selected point and
 * hour. Everything else is deliberate and should not be redesigned.
 *
 * Data it expects, from Supabase:
 *   coefficients  Record<string, number> for the selected species, keyed on
 *                 `term` from the model_coefficients table
 *   hotspots      the 192 rows of the hotspots table
 *
 * Scoring comes from ./score — do not reimplement it here.
 */
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { CircleMarker, MapContainer, Rectangle, TileLayer, Tooltip }
  from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { EXTENT, inCoverage, modelledWeeks, score, unavailable, weekOf }
  from "./score";

type Coef = Record<string, number>;

export interface Hotspot {
  loc_id: string;
  name: string;
  latitude: number;
  longitude: number;
  n_checklists: number;
}

interface HourWeather {
  windMax: number; tempMean: number; precipSum: number; cloudMean: number;
}

const pad = (n: number) => String(n).padStart(2, "0");
const HOURS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

/** Pale magenta to deep purple. A street basemap contains no purple, and
 *  lightness falls along it so the order survives greyscale. */
function ramp(t: number): string {
  const a = [252, 217, 240], b = [74, 12, 85];
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

/**
 * Open-Meteo's forecast for one point, aggregated over the hours an outing
 * spans. Wind is the MAX rather than the mean: a gust that masks birdsong
 * matters more than the average across two hours.
 */
async function fetchWeather(
  lat: number, lon: number, day: string, hour: number, durationHrs: number,
): Promise<HourWeather | null> {
  const url = "https://api.open-meteo.com/v1/forecast"
    + `?latitude=${lat}&longitude=${lon}`
    + "&hourly=temperature_2m,precipitation,wind_speed_10m,cloud_cover"
    + "&forecast_days=16&timezone=auto";
  const r = await fetch(url);
  if (!r.ok) return null;
  const h = (await r.json()).hourly;

  const want = new Set<string>();
  for (let i = 0; i < Math.max(1, Math.ceil(durationHrs)); i++) {
    want.add(`${day}T${pad(hour + i)}:00`);
  }
  const idx = h.time.flatMap((t: string, i: number) => want.has(t) ? [i] : []);
  if (!idx.length) return null;          // outside the forecast horizon

  const pick = (k: string) => idx.map((i: number) => h[k][i]);
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  return {
    windMax: Math.max(...pick("wind_speed_10m")),
    tempMean: mean(pick("temperature_2m")),
    precipSum: pick("precipitation").reduce((x: number, y: number) => x + y, 0),
    cloudMean: mean(pick("cloud_cover")),
  };
}

export function ForecastPanel(
  { coefficients, hotspots, speciesName }:
  { coefficients: Coef; hotspots: Hotspot[]; speciesName: string },
) {
  const today = new Date().toISOString().slice(0, 10);
  const [day, setDay] = useState(today);
  const [hour, setHour] = useState(8);
  const [durationHrs, setDuration] = useState(1);
  const [wx, setWx] = useState<HourWeather | null>(null);
  const [loading, setLoading] = useState(true);

  // one fetch per (day, hour, duration); the whole county is a single 0.25 deg
  // neighbourhood, so fetching per hotspot would be 192 identical calls
  useEffect(() => {
    let live = true;
    setLoading(true);
    const [cLat, cLon] = [(EXTENT.latMin + EXTENT.latMax) / 2,
                         (EXTENT.lonMin + EXTENT.lonMax) / 2];
    fetchWeather(cLat, cLon, day, hour, durationHrs).then(w => {
      if (live) { setWx(w); setLoading(false); }
    });
    return () => { live = false; };
  }, [day, hour, durationHrs]);

  const date = useMemo(() => new Date(day + "T12:00:00Z"), [day]);
  const weeks = useMemo(() => modelledWeeks(coefficients), [coefficients]);
  const wk = weekOf(date);
  const why = unavailable(coefficients, date, hour);

  const scored = useMemo(() => {
    if (why || !wx) return [];
    const o = { durationHrs, distanceKm: 1, observers: 1, traveling: true };
    return hotspots
      .map(h => ({ ...h, p: score(coefficients, date, hour, o, wx, h.loc_id) }))
      .filter((h): h is Hotspot & { p: number } => h.p !== null)
      .sort((a, b) => b.p - a.p);
  }, [coefficients, hotspots, date, hour, durationHrs, wx, why]);

  const hi = scored[0]?.p ?? 0;
  const lo = scored[scored.length - 1]?.p ?? 0;

  return (
    <div className="flex h-full">
      <aside className="w-[330px] shrink-0 overflow-y-auto border-r p-4 space-y-3">
        <div>
          <h2 className="text-base font-semibold">{speciesName}</h2>
          <p className="text-xs text-muted-foreground">
            Allegheny County, Pennsylvania
          </p>
        </div>

        <Field label="Date">
          <input type="date" value={day} min={today} onChange={e => setDay(e.target.value)} />
        </Field>
        <Field label="Hour">
          <select value={hour} onChange={e => setHour(Number(e.target.value))}>
            {HOURS.map(h => <option key={h} value={h}>{pad(h)}:00</option>)}
          </select>
        </Field>
        <Field label="How long">
          <select value={durationHrs} onChange={e => setDuration(Number(e.target.value))}>
            <option value={0.25}>15 minutes</option>
            <option value={0.5}>30 minutes</option>
            <option value={1}>1 hour</option>
            <option value={2}>2 hours</option>
            <option value={3}>3 hours</option>
          </select>
        </Field>

        {/* The coverage bar sits above the result on purpose. The user has to
            see how much of the year this model cannot speak to before reading
            anything into a number. */}
        <div>
          <Label>Weeks the model covers</Label>
          <div className="flex gap-px">
            {Array.from({ length: 53 }, (_, i) => (
              <i key={i}
                 className={`h-4 flex-1 rounded-[1px] ${
                   weeks.includes(i) ? "bg-teal-600" : "bg-slate-200"} ${
                   i === wk ? "outline outline-2 outline-amber-600" : ""}`} />
            ))}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {weeks.length} of 53 weeks fitted · this date is week {wk}
          </p>
        </div>

        {why === "week" && (
          <Note warn>
            <b>No data for week {wk}.</b> The model was fitted on scraped days
            that do not cover this part of the year. Nothing is scored — a
            number here would mean dropping seasonality out of the calculation.
          </Note>
        )}
        {why === "hour" && (
          <Note warn>
            <b>Too few checklists at {pad(hour)}:00.</b> Support is{" "}
            {coefficients[`support_${pad(hour)}`] ?? 0}; the floor is 50.
          </Note>
        )}
        {!why && loading && <Note>Fetching the forecast…</Note>}
        {!why && !loading && !wx && (
          <Note warn>
            <b>Beyond the forecast horizon.</b> Weather is available 16 days
            ahead; pick a nearer date.
          </Note>
        )}
        {!why && wx && scored.length > 0 && (
          <>
            <Note>
              {scored.length} hotspots scored. Range{" "}
              {(lo * 100).toFixed(1)}% to {(hi * 100).toFixed(1)}%.
              <span className="block text-muted-foreground">
                Forecast: wind {wx.windMax.toFixed(0)} km/h,{" "}
                {wx.tempMean.toFixed(0)}°C, {wx.precipSum.toFixed(1)} mm rain,{" "}
                {wx.cloudMean.toFixed(0)}% cloud
              </span>
            </Note>

            <div>
              <Label>Best hotspots</Label>
              <table className="w-full text-xs">
                <tbody>
                  {scored.slice(0, 12).map(h => (
                    <tr key={h.loc_id} className="border-b">
                      <td className="py-1">{h.name}</td>
                      <td className="py-1 text-right font-semibold tabular-nums text-teal-700">
                        {(h.p * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </aside>

      <MapContainer
        bounds={[[EXTENT.latMin, EXTENT.lonMin], [EXTENT.latMax, EXTENT.lonMax]]}
        className="flex-1"
        style={{ height: "100%" }}   /* Leaflet renders 0px tall without this */
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          subdomains="abcd"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>'
        />
        {/* the rectangle inCoverage() tests, not the county's true outline —
            drawing the real boundary would promise forecasts outside what the
            code allows and deny them inside it */}
        <Rectangle
          bounds={[[EXTENT.latMin, EXTENT.lonMin], [EXTENT.latMax, EXTENT.lonMax]]}
          pathOptions={{ color: "#8a9a97", weight: 1, fill: false, dashArray: "5 5" }} />
        {scored.map(h => {
          const t = hi === lo ? 0.5 : (h.p - lo) / (hi - lo);
          return (
            <CircleMarker key={h.loc_id}
              center={[h.latitude, h.longitude]}
              radius={4 + 7 * t}
              pathOptions={{ fillColor: ramp(t), color: "#fff", weight: 1.6,
                             fillOpacity: 0.95 }}>
              <Tooltip direction="top">
                <b>{h.name}</b><br />{(h.p * 100).toFixed(1)}%<br />
                <span className="text-muted-foreground">
                  {h.n_checklists} checklists in training data
                </span>
              </Tooltip>
            </CircleMarker>
          );
        })}
      </MapContainer>
    </div>
  );
}

const Label = ({ children }: { children: ReactNode }) => (
  <span className="block text-[11px] uppercase tracking-wide text-muted-foreground">
    {children}
  </span>
);

const Field = ({ label, children }:
               { label: string; children: ReactNode }) => (
  <div><Label>{label}</Label>{children}</div>
);

const Note = ({ children, warn }:
              { children: ReactNode; warn?: boolean }) => (
  <div className={`rounded-r border-l-2 p-2 text-xs ${
    warn ? "border-amber-600 bg-amber-50" : "border-teal-600 bg-teal-50"}`}>
    {children}
  </div>
);
