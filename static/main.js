// UrbanFlow AI — dashboard on a real OpenStreetMap map of Hyderabad.
// Flask returns real road routes (OpenStreetMap roads), turn-by-turn directions and ML predictions.

const $ = (id) => document.getElementById(id);

const MAP_CENTER = [17.41, 78.45];
let map, nodes = {};
let routeLayer;            // everything drawn for the current result
let markerLayer;           // start / end markers
let nodeLayer;             // small dots for the known locations
let currentSteps = [];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}
function escapeClass(value) {
  return String(value ?? "info").toLowerCase().replace(/[^a-z0-9_-]/g, "");
}
function setText(id, value, fallback = "—") {
  const el = $(id);
  if (el) el.textContent = value === undefined || value === null || value === "" ? fallback : String(value);
}

function mapIcon(label, color) {
  return L.divIcon({
    className: "custom-map-marker",
    html: `<span style="background:${color}">${label}</span>`,
    iconSize: [27, 27],
    iconAnchor: [13, 13]
  });
}

// ---------- map ----------
function initMap() {
  const el = $("map");
  if (!window.L) {
    el.innerHTML = '<div style="padding:20px;color:#111">Map library could not load. Check your internet connection.</div>';
    return;
  }
  map = L.map("map", { zoomControl: true }).setView(MAP_CENTER, 12);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  nodeLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  markerLayer = L.layerGroup().addTo(map);
}

function drawNodes() {
  nodeLayer.clearLayers();
  Object.entries(nodes).forEach(([id, n]) => {
    L.circleMarker([n.lat, n.lng], { radius: 5, color: "#111", weight: 1.5, fillColor: "#8a63d2", fillOpacity: 1 })
      .addTo(nodeLayer).bindTooltip(`${id} · ${n.name}`);
  });
}

function drawEndpoints(fit) {
  if (!map) return;
  markerLayer.clearLayers();
  const s = nodes[$("startNode").value], e = nodes[$("endNode").value];
  if (!s || !e) return;
  L.marker([s.lat, s.lng], { icon: mapIcon("+", "#b8d8f8") }).addTo(markerLayer)
    .bindPopup(`<b>Start</b><br>${escapeHtml(s.name)}`);
  L.marker([e.lat, e.lng], { icon: mapIcon("H", "#65d7a1") }).addTo(markerLayer)
    .bindPopup(`<b>End</b><br>${escapeHtml(e.name)}`);
  if (fit) map.fitBounds(L.latLngBounds([[s.lat, s.lng], [e.lat, e.lng]]).pad(0.35));
}

function clearRoute() {
  routeLayer.clearLayers();
  currentSteps = [];
}

function drawResult(data) {
  routeLayer.clearLayers();
  const bounds = [];

  (data.alternate_routes_coordinates || []).forEach((coords) => {
    if (coords.length < 2) return;
    L.polyline(coords, { color: "#4d8fe8", weight: 5, opacity: 0.85, dashArray: "8 8" }).addTo(routeLayer);
    bounds.push(...coords);
  });

  const main = data.route_coordinates || [];
  if (main.length >= 2) {
    L.polyline(main, { color: "#ffffff", weight: 10, opacity: 0.95 }).addTo(routeLayer);   // casing
    L.polyline(main, { color: "#1a9f5c", weight: 6, opacity: 1 }).addTo(routeLayer);
    bounds.push(...main);
  }

  currentSteps = data.directions || [];
  currentSteps.forEach((step, i) => {
    if (!step.is_turn) return;
    L.circleMarker(step.location, { radius: 4.5, color: "#111", weight: 1.5, fillColor: "#fff", fillOpacity: 1 })
      .addTo(routeLayer)
      .bindTooltip(`${step.instruction} (${step.distance_text})`);
  });

  if (data.incident_coordinates) {
    L.marker(data.incident_coordinates, { icon: mapIcon("!", "#f0525b") }).addTo(routeLayer)
      .bindPopup("<b>Crash</b>").openPopup();
    bounds.push(data.incident_coordinates);
  }
  if (bounds.length) map.fitBounds(L.latLngBounds(bounds).pad(0.12));
}

// ---------- panels ----------
function renderDirections(data) {
  const list = $("directionsList");
  const steps = data.directions || [];
  const s = data.route_summary;
  setText("directionsSummary", s ? `${s.distance_text} · ${s.turns} turns` : "—");
  if (!steps.length) {
    list.innerHTML = '<div class="empty-hint">No route.</div>';
    return;
  }
  list.innerHTML = steps.map((st, i) =>
    `<div class="step-row" data-step="${i}"><span class="step-icon">${escapeHtml(st.icon)}</span>` +
    `<div class="step-text"><b>${escapeHtml(st.instruction)}</b></div>` +
    `<span class="step-dist">${escapeHtml(st.distance_text)}</span></div>`
  ).join("");
}

function renderRoutes(data) {
  const routes = data.routes || [];
  $("routesList").innerHTML = routes.length ? routes.map((r, i) =>
    `<div class="route-row ${r.selected ? "selected" : ""}"><span class="route-index">0${i + 1}</span>` +
    `<div class="route-name"><b>${escapeHtml(r.name)}</b><small>${escapeHtml(r.distance)}</small></div>` +
    `<strong>${escapeHtml(r.estimated_time)}</strong>${r.selected ? "<em>SELECTED</em>" : ""}</div>`
  ).join("") : '<div class="route-row"><span class="route-index">—</span><div class="route-name"><b>None</b><small>—</small></div><strong>—</strong></div>';
}

function renderLog(data) {
  const logs = data.agent_log || [];
  $("agentLog").innerHTML = logs.map((l) =>
    `<div class="log-row ${escapeClass(l.type)}"><time>${escapeHtml(l.time)}</time><b>${escapeHtml(l.agent)}</b><span>${escapeHtml(l.message)}</span></div>`
  ).join("");
}

function renderPrediction(p) {
  const body = $("predictBody");
  if (!p) return;
  setText("predictMeta", `${p.time_label} · ${p.weather}`);
  const level = String(p.congestion_level).toLowerCase();
  const cls = level === "high" ? "bad" : level === "medium" ? "warn" : "good";
  const max = Math.max(...p.curve.map((c) => c.score), 1);
  const bars = p.curve.map((c) => {
    const tag = c.hour === p.peak.hour ? "peak" : c.hour === p.best.hour ? "best" : c.label === p.time_label ? "now" : "";
    return `<div class="bar ${tag}" title="${c.label} · congestion ${c.score}" style="height:${Math.max((c.score / max) * 100, 2)}%"></div>`;
  }).join("");
  body.innerHTML =
    `<div class="pred-grid">` +
    `<div class="pred-card ${cls}"><label>CONGESTION</label><strong>${p.congestion_score} · ${escapeHtml(level.toUpperCase())}</strong></div>` +
    `<div class="pred-card"><label>DISTANCE</label><strong>${escapeHtml(p.route.distance_text)}</strong></div>` +
    `<div class="pred-card"><label>ETA</label><strong>${escapeHtml(p.route.eta_text)}</strong></div>` +
    `<div class="pred-card"><label>TURNS</label><strong>${p.route.turns}</strong></div>` +
    `<div class="pred-card good"><label>BEST TIME</label><strong>${escapeHtml(p.best.label)} · ${escapeHtml(p.best_eta)}</strong></div>` +
    `<div class="pred-card bad"><label>WORST TIME</label><strong>${escapeHtml(p.peak.label)}</strong></div>` +
    `</div>` +
    `<div class="pred-title">CONGESTION BY HOUR</div>` +
    `<div class="bars">${bars}</div>` +
    `<div class="bar-labels"><span>00</span><span>06</span><span>12</span><span>18</span><span>23</span></div>` +
    `<div class="bar-legend"><span><i style="background:var(--yellow)"></i>now</span><span><i style="background:#24b86b"></i>best</span><span><i style="background:#f0525b"></i>worst</span></div>`;
}

function renderResponse(data) {
  const congestion = Number(data.congestion_score);
  if (Number.isFinite(congestion)) {
    setText("congestion", Math.round(congestion));
    setText("severity", congestion >= 70 ? "HIGH" : congestion >= 40 ? "MODERATE" : "LOW");
  }
  setText("incidents", data.active_incidents ?? 0);
  setText("vehicles", data.vehicle_count);
  setText("decisionHeading", data.decision?.heading);
  setText("decisionText", data.decision?.text);
  setText("agentName", data.decision_by);
  setText("confidence", data.confidence == null ? "—" : `${(Number(data.confidence) * 100).toFixed(1)}%`);
  setText("timeSaved", data.eta_text);
  setText("mapTime", new Date().toLocaleTimeString());

  renderLog(data);
  renderRoutes(data);
  renderDirections(data);
  if (data.prediction) renderPrediction(data.prediction);
  if (map) drawResult(data);
}

// ---------- requests ----------
function collectInputs() {
  return {
    start_node: $("startNode").value,
    end_node: $("endNode").value,
    weather: Number($("weather").value),
    time: Number($("timeOfDay").value),
    vehicle_count: Number($("vehicleCount").value)
  };
}

async function callBackend(url, extra, button, busyText, idleText) {
  const buttons = [$("simulateBtn"), $("predictBtn")];
  buttons.forEach((b) => (b.disabled = true));
  button.textContent = busyText;
  setText("mapTime", "Loading…");
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ ...collectInputs(), ...extra })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.message ?? data.error ?? `Server returned HTTP ${response.status}`);
    renderResponse(data);
  } catch (error) {
    console.error("UrbanFlow request failed:", error);
    setText("mapTime", "Failed");
    $("agentLog").innerHTML = `<div class="log-row alert"><time>ERROR</time><b>API</b><span>${escapeHtml(error.message)}</span></div>`;
    setText("decisionHeading", "FAILED");
    setText("decisionText", error.message);
  } finally {
    buttons.forEach((b) => (b.disabled = false));
    button.textContent = idleText;
  }
}

const runSimulation = () => callBackend("/simulate_incident", { trigger_crash: true }, $("simulateBtn"),
  "WAIT...", "SIMULATE CRASH");
const runPrediction = () => callBackend("/predict_all", {}, $("predictBtn"),
  "WAIT...", "PREDICT");

// ---------- UI wiring ----------
function updateSliderLabels() {
  const time = Number($("timeOfDay").value);
  const hours = Math.floor(time);
  const minutes = Math.round((time - hours) * 60);
  $("timeValue").textContent = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
  $("vehicleValue").textContent = $("vehicleCount").value;
}

async function loadNodes() {
  try {
    const res = await fetch("/api/nodes");
    nodes = await res.json();
  } catch (e) {
    setText("mapTime", "No locations");
    return;
  }
  const options = Object.entries(nodes).map(([id, n]) => `<option value="${id}">${id} · ${escapeHtml(n.name)}</option>`).join("");
  $("startNode").innerHTML = options;
  $("endNode").innerHTML = options;
  $("startNode").value = "A";
  $("endNode").value = "J";
  if (map) { drawNodes(); drawEndpoints(true); }
}

function onEndpointChange() {
  clearRoute();
  drawEndpoints(true);
}

$("simulateBtn").addEventListener("click", runSimulation);
$("predictBtn").addEventListener("click", runPrediction);
$("themeToggle").addEventListener("click", () => {
  document.body.classList.toggle("night-page");
  $("themeToggle").textContent = document.body.classList.contains("night-page") ? "☼" : "☾";
});
$("startNode").addEventListener("change", onEndpointChange);
$("endNode").addEventListener("change", onEndpointChange);
$("timeOfDay").addEventListener("input", updateSliderLabels);
$("vehicleCount").addEventListener("input", updateSliderLabels);
$("directionsList").addEventListener("click", (event) => {
  const row = event.target.closest(".step-row");
  if (!row || !map) return;
  const step = currentSteps[Number(row.dataset.step)];
  if (step) map.setView(step.location, 17);
});

initMap();
updateSliderLabels();
$("agentLog").innerHTML = '<div class="log-row info"><time></time><b>System</b><span>Ready.</span></div>';
loadNodes();
