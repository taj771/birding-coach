"""Build a standalone HTML preview of the forecast map.

    python pipeline/make_preview.py   ->  data/preview.html

Open it in a browser. No build step, no npm, no server — the coefficients and
the hotspots are embedded, Leaflet comes from a CDN and the weather from
Open-Meteo, neither of which needs a key.

The point is to see what the map looks like against the real model before
asking Lovable to build it: which hotspots stand out, how thin the probability
spread actually is, and how much of the year is greyed out. Cheaper to find out
here than through a chat interface that charges per attempt.

The scoring is a transcription of serve/score.ts. It is duplicated on purpose —
this file is a throwaway preview, and wiring a build step to share one module
would cost more than it saves. If score.ts changes, this follows.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
COEF = ROOT / "data" / "model_coefficients.json"
OUT = ROOT / "data" / "preview.html"

TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Birding Coach — map preview</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root { --ink:#1a2b28; --accent:#00897A; --muted:#6b7b78; --line:#dbe4e2; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;
         color:var(--ink); display:flex; height:100vh; }
  #panel { width:330px; flex:none; padding:18px; overflow-y:auto;
           border-right:1px solid var(--line); }
  #map { flex:1; }
  h1 { font-size:17px; margin:0 0 2px; }
  .sub { color:var(--muted); font-size:12px; margin-bottom:16px; }
  label { display:block; font-size:11px; text-transform:uppercase;
          letter-spacing:.05em; color:var(--muted); margin:12px 0 4px; }
  select, input { width:100%; padding:7px 8px; border:1px solid var(--line);
                  border-radius:5px; font:inherit; background:#fff; }
  .note { background:#f4f8f7; border-left:2px solid var(--accent);
          padding:9px 11px; font-size:12.5px; margin:14px 0; border-radius:0 4px 4px 0; }
  .warn { background:#fdf6ec; border-left-color:#BE7A12; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:6px; }
  td { padding:3px 0; border-bottom:1px solid var(--line); }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums;
                  color:var(--accent); font-weight:600; }
  .weeks { display:flex; gap:1px; margin:6px 0 2px; }
  .weeks i { flex:1; height:16px; background:var(--line); border-radius:1px; }
  .weeks i.on { background:var(--accent); }
  .weeks i.now { outline:2px solid #BE7A12; outline-offset:1px; }
  .wx { display:grid; grid-template-columns:auto 1fr 32px; gap:4px 8px;
        align-items:center; font-size:12px; }
  .wx input { padding:0; }
  .wx b { text-align:right; font-variant-numeric:tabular-nums;
          font-weight:600; color:var(--accent); }
  .legend { display:flex; align-items:center; gap:6px; font-size:11px;
            color:var(--muted); margin-top:10px; }
  .ramp { flex:1; height:9px; border-radius:5px;
          background:linear-gradient(90deg,#e8f2f0,#00897A,#00443d); }
</style>

<div id="panel">
  <h1>Birding Coach</h1>
  <div class="sub">map preview &middot; real coefficients, real hotspots</div>

  <label>Species</label>
  <select id="sp"></select>

  <label>Date</label>
  <input type="date" id="date">

  <label>Hour</label>
  <select id="hr"></select>

  <label>How long</label>
  <select id="dur">
    <option value="0.25">15 minutes</option>
    <option value="0.5">30 minutes</option>
    <option value="1" selected>1 hour</option>
    <option value="2">2 hours</option>
    <option value="3">3 hours</option>
  </select>

  <label>Weather <span class="sub" style="text-transform:none">— the app fills
    these from the forecast</span></label>
  <div class="wx">
    <span>Wind</span><input type="range" id="wind" min="0" max="36" value="12">
      <b id="windv">12</b>
    <span>Temp</span><input type="range" id="temp" min="-5" max="32" value="18">
      <b id="tempv">18</b>
    <span>Rain</span><input type="range" id="rain" min="0" max="18" step="0.5" value="0">
      <b id="rainv">0</b>
    <span>Cloud</span><input type="range" id="cloud" min="0" max="100" value="60">
      <b id="cloudv">60</b>
  </div>

  <div id="coverage"></div>
  <div id="wxeffect"></div>

  <label>Weeks the model covers</label>
  <div class="weeks" id="weeks"></div>
  <div class="sub" id="weeklabel"></div>

  <div class="legend"><span>low</span><span class="ramp"></span><span>high</span></div>

  <label>Best hotspots</label>
  <table id="top"></table>
</div>
<div id="map"></div>

<script>
const COEF = __COEF__;
const HOTSPOTS = __HOTSPOTS__;
const EXTENT = { latMin:40.19, latMax:40.68, lonMin:-80.34, lonMax:-79.69 };

// --- transcription of serve/score.ts ------------------------------------
const pad = n => String(n).padStart(2, "0");
function weekOf(d) {
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const doy = Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(),
                                   d.getUTCDate()) - start) / 86400000);
  return Math.floor(doy / 7);
}
function unavailable(c, date, hour, minSupport = 50) {
  if ((c[`support_${pad(hour)}`] ?? 0) < minSupport) return "hour";
  if (c[`week_${pad(weekOf(date))}`] === undefined) return "week";
  return null;
}
function modelledWeeks(c) {
  return Object.keys(c).filter(k => k.startsWith("week_"))
                       .map(k => Number(k.slice(5))).sort((a,b) => a-b);
}
function score(c, date, hour, o, w, locId) {
  if (unavailable(c, date, hour)) return null;
  const site = (locId && c[`site_${locId}`] !== undefined)
    ? c[`site_${locId}`] : (c.site_PERSONAL ?? 0);
  const logit = (c.intercept ?? 0)
    + (c[`hour_${pad(hour)}`] ?? 0)
    + (c[`week_${pad(weekOf(date))}`] ?? 0)
    + site
    + (c.log_dur ?? 0) * Math.log(o.durationHrs)
    + (c.log_dist ?? 0) * Math.log1p(o.distanceKm)
    + (c.n_observers ?? 0) * o.observers
    + (c.traveling ?? 0) * (o.traveling ? 1 : 0)
    + (c.wind_max ?? 0) * w.windMax
    + (c.temp_mean ?? 0) * w.tempMean
    + (c.precip_sum ?? 0) * w.precipSum
    + (c.cloud_mean ?? 0) * w.cloudMean;
  return 1 / (1 + Math.exp(-logit));
}
// ------------------------------------------------------------------------

const map = L.map("map").fitBounds([[EXTENT.latMin, EXTENT.lonMin],
                                    [EXTENT.latMax, EXTENT.lonMax]]);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  maxZoom: 18,
}).addTo(map);

// the rectangle inCoverage() actually tests, not the county's true outline
L.rectangle([[EXTENT.latMin, EXTENT.lonMin], [EXTENT.latMax, EXTENT.lonMax]],
            { color:"#00897A", weight:1, fillOpacity:0.03, dashArray:"4 4" })
 .addTo(map);

const layer = L.layerGroup().addTo(map);

// In the app these come from Open-Meteo's 7-day hourly forecast. Here they are
// sliders, because every forecastable date falls in a week the model has no
// term for — a live fetch would only ever show the gap panel. Sliders also make
// the size of the weather effect visible, which a fetch would not.
const WX_IDS = { windMax:"wind", tempMean:"temp", precipSum:"rain", cloudMean:"cloud" };
const weather = () => Object.fromEntries(Object.entries(WX_IDS).map(
  ([k, id]) => [k, Number(document.getElementById(id).value)]));

// what the model was fitted around; the reference the weather effect is measured against
const WX_MEDIAN = { windMax:12, tempMean:18, precipSum:0, cloudMean:60 };

const spSel = document.getElementById("sp");
Object.keys(COEF).sort().forEach(k => {
  const o = document.createElement("option");
  o.value = k; o.textContent = `${COEF[k].name} (${k})`;
  spSel.appendChild(o);
});
spSel.value = "norpar";

const hrSel = document.getElementById("hr");
for (let h = 6; h <= 20; h++) {
  const o = document.createElement("option");
  o.value = h; o.textContent = `${pad(h)}:00`;
  hrSel.appendChild(o);
}
hrSel.value = 8;

const dateEl = document.getElementById("date");
dateEl.value = "2026-05-14";       // a week the model covers

function ramp(t) {                  // single hue, light -> dark
  const a = [232,242,240], b = [0,68,61];
  return `rgb(${a.map((v,i) => Math.round(v + (b[i]-v)*t)).join(",")})`;
}

function draw() {
  const c = COEF[spSel.value].terms;
  const date = new Date(dateEl.value + "T12:00:00Z");
  const hour = Number(hrSel.value);
  const o = { durationHrs:Number(document.getElementById("dur").value),
              distanceKm:1, observers:1, traveling:true };

  const weeks = modelledWeeks(c);
  const wk = weekOf(date);
  const wEl = document.getElementById("weeks");
  wEl.innerHTML = "";
  for (let i = 0; i <= 52; i++) {
    const s = document.createElement("i");
    if (weeks.includes(i)) s.className = "on";
    if (i === wk) s.className += " now";
    wEl.appendChild(s);
  }
  document.getElementById("weeklabel").textContent =
    `${weeks.length} of 53 weeks fitted · this date is week ${wk}`;

  const why = unavailable(c, date, hour);
  const cov = document.getElementById("coverage");
  layer.clearLayers();
  document.getElementById("top").innerHTML = "";

  if (why === "week") {
    cov.innerHTML = `<div class="note warn"><b>No data for week ${wk}.</b><br>
      The model was fitted on 48 scraped days, so most of the calendar has no
      seasonal term. Nothing is scored — showing a number here would mean
      dropping seasonality out of the sum.</div>`;
    return;
  }
  if (why === "hour") {
    cov.innerHTML = `<div class="note warn"><b>Too few checklists at
      ${pad(hour)}:00.</b><br>Support is ${c[`support_${pad(hour)}`] ?? 0};
      the floor is 50.</div>`;
    return;
  }

  const wx = weather();
  const scored = HOTSPOTS.map(h => ({
    ...h, p: score(c, date, hour, o, wx, h.loc_id),
  })).filter(h => h.p !== null).sort((a,b) => b.p - a.p);

  const hi = scored[0].p, lo = scored[scored.length-1].p;
  cov.innerHTML = `<div class="note">${scored.length} hotspots scored.
    Range ${(lo*100).toFixed(1)}% to ${(hi*100).toFixed(1)}%.</div>`;

  // how much the weather sliders are worth, at the best site, against median
  const best = scored[0];
  const atMedian = score(c, date, hour, o, WX_MEDIAN, best.loc_id);
  const delta = (best.p - atMedian) * 100;
  document.getElementById("wxeffect").innerHTML =
    `<div class="note">Weather is worth <b>${delta >= 0 ? "+" : ""}${delta.toFixed(1)}
     points</b> at ${best.name}, against median conditions
     (wind 12, 18&deg;C, no rain, 60% cloud).<br>
     <span style="color:var(--muted)">Same outing at median weather:
     ${(atMedian*100).toFixed(1)}% &middot; now: ${(best.p*100).toFixed(1)}%</span></div>`;

  scored.forEach(h => {
    const t = hi === lo ? 0.5 : (h.p - lo) / (hi - lo);
    L.circleMarker([h.latitude, h.longitude], {
      radius: 4 + 7*t, fillColor: ramp(t), color:"#fff",
      weight: 1, fillOpacity: 0.9,
    }).bindTooltip(
      `<b>${h.name}</b><br>${(h.p*100).toFixed(1)}%<br>
       <span style="color:#6b7b78">${h.n_checklists} checklists in training data</span>`,
      { direction:"top" }).addTo(layer);
  });

  document.getElementById("top").innerHTML = scored.slice(0, 12).map(h =>
    `<tr><td>${h.name}</td><td>${(h.p*100).toFixed(1)}%</td></tr>`).join("");
}

const inputs = [spSel, dateEl, hrSel, document.getElementById("dur"),
                ...Object.values(WX_IDS).map(id => document.getElementById(id))];
inputs.forEach(el => el.addEventListener("input", () => {
  for (const id of Object.values(WX_IDS)) {
    document.getElementById(id + "v").textContent =
      document.getElementById(id).value;
  }
  draw();
}));
draw();
</script>
"""


def load_hotspots():
    """Parse the generated SQL back into rows — one source of truth."""
    import re
    sql = (ROOT / "data" / "hotspots.sql").read_text()
    rows = re.findall(r"\('(.+?)', '(.*?)', ([-\d.]+), ([-\d.]+), (\d+)\)", sql)
    return [{"loc_id": a, "name": b.replace("''", "'"),
             "latitude": float(c), "longitude": float(d), "n_checklists": int(e)}
            for a, b, c, d, e in rows]


def flat(m):
    """Nested model dict -> the flat term lookup score() expects."""
    out = {k: v for k, v in m.items() if not isinstance(v, dict)}
    for h, v in m.get("hour", {}).items():
        out[f"hour_{int(h):02d}"] = v
    for w, v in m.get("week", {}).items():
        out[f"week_{int(w):02d}"] = v
    for s, v in m.get("site", {}).items():
        out[f"site_{s}"] = v
    for h, n in m.get("hour_support", {}).items():
        out[f"support_{int(h):02d}"] = n
    return out


if __name__ == "__main__":
    models = json.loads(COEF.read_text())
    hotspots = load_hotspots()

    names = {}
    sp_file = ROOT / "data" / "species_names.json"
    if sp_file.exists():
        names = json.loads(sp_file.read_text())

    coef = {sp: {"name": names.get(sp, sp), "terms": flat(m)}
            for sp, m in models.items()}

    html = (TEMPLATE
            .replace("__COEF__", json.dumps(coef, separators=(",", ":")))
            .replace("__HOTSPOTS__", json.dumps(hotspots, separators=(",", ":"))))
    OUT.write_text(html)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, "
          f"{len(coef)} species, {len(hotspots)} hotspots)")
    print(f"open it:  open {OUT}")
