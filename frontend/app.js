/* GeoData AI · Fire severity map. Vanilla JS, Leaflet and anime.js against the
   FastAPI service on the same origin. Motion marks state changes only and is
   disabled under prefers-reduced-motion. Colours are read from CSS custom
   properties so palettes and dark mode apply to the map and chart too. */
(function () {
  "use strict";

  const SOURCE_LABELS = { modis: "MODIS", viirs_j1: "VIIRS NOAA‑20", viirs_suomi: "VIIRS Suomi NPP" };
  const SOURCE_VARS = { modis: "--src-1", viirs_j1: "--src-2", viirs_suomi: "--src-3" };
  const MOTION = !window.matchMedia("(prefers-reduced-motion: reduce)").matches && typeof anime === "function";
  const DEFAULT_CAPTION = "Detections per day. Severe share in red.";
  const el = (id) => document.getElementById(id);
  const fmt = new Intl.NumberFormat("en-US");
  const dateLabel = (iso) => new Date(iso + "T00:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = { health: null, layer: null, pin: null, features: [], stats: null, loading: false, counters: {}, chartRows: [], chartProgress: 1, lastFilters: null, lastProb: 0, baseName: null };

  // ---------------------------------------------------------------- motion
  function tween(params) {
    if (MOTION) return anime(params);
    // Reduced motion: jump straight to the end state.
    const { targets, update, complete } = params;
    const end = {};
    Object.keys(params).forEach((key) => {
      if (["targets", "duration", "delay", "easing", "update", "complete", "loop", "direction", "round"].includes(key)) return;
      const v = params[key];
      end[key] = Array.isArray(v) ? v[v.length - 1] : v;
    });
    const list = typeof targets === "string" ? document.querySelectorAll(targets) : (targets.length != null && !(targets instanceof Element)) ? targets : [targets];
    Array.from(list).forEach((t) => {
      if (!(t instanceof Element)) { Object.assign(t, end); return; }
      const tf = [];
      Object.entries(end).forEach(([k, v]) => {
        if (k === "opacity") t.style.opacity = v;
        else if (k === "translateX" || k === "translateY") tf.push(`${k}(${typeof v === "number" ? v + "px" : v})`);
        else if (k === "scale" || k === "scaleX") tf.push(`${k}(${v})`);
        else if (k === "width") t.style.width = typeof v === "number" ? v + "px" : v;
      });
      if (tf.length) t.style.transform = tf.join(" ");
    });
    if (update) update();
    if (complete) complete();
    return null;
  }
  function countTo(id, value, formatter) {
    const node = el(id);
    const counter = state.counters[id] || (state.counters[id] = { value: 0 });
    tween({ targets: counter, value: value || 0, duration: 700, easing: "easeOutExpo", update: () => { node.textContent = formatter(counter.value); } });
  }
  function enterSections() {
    document.querySelectorAll("[data-enter]").forEach((s) => { s.style.opacity = "0"; });
    tween({ targets: "[data-enter]", opacity: [0, 1], translateY: [12, 0], duration: 480, delay: MOTION ? anime.stagger(70, { start: 60 }) : 0, easing: "easeOutCubic" });
  }

  // ------------------------------------------------------------ segmented
  function initSegmented(container) {
    const thumb = container.querySelector(".segmented-thumb");
    const move = (animate) => {
      const checked = container.querySelector("input:checked");
      const label = (checked || container.querySelector("input")).closest("label");
      if (!label.offsetWidth) return; // hidden (inside a closed popover / details)
      const x = label.offsetLeft - 3, w = label.offsetWidth;
      if (!animate || !MOTION) { thumb.style.transform = `translateX(${x}px)`; thumb.style.width = `${w}px`; return; }
      anime({ targets: thumb, translateX: x, width: w, duration: 320, easing: "spring(1, 90, 14, 0)" });
    };
    container.addEventListener("change", () => move(true));
    requestAnimationFrame(() => move(false));
    window.addEventListener("resize", () => move(false));
    return { refresh: () => move(false) };
  }
  const segments = Array.from(document.querySelectorAll("[data-segmented]")).map(initSegmented);
  const refreshSegments = () => segments.forEach((s) => s.refresh());
  document.querySelectorAll("details").forEach((d) => d.addEventListener("toggle", () => requestAnimationFrame(refreshSegments)));

  // ---------------------------------------------------------------- theme
  const THEME_KEY = "geodata-theme";
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
  function readTheme() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(THEME_KEY) || "{}"); } catch (_) { /* storage unavailable */ }
    // ?palette=forest&mode=dark lets a look be shared by link.
    const q = new URLSearchParams(location.search);
    if (q.get("palette")) saved.palette = q.get("palette");
    if (q.get("mode")) saved.appearance = q.get("mode");
    return saved;
  }
  function isDark() { const t = document.documentElement.getAttribute("data-theme"); return t ? t === "dark" : darkQuery.matches; }
  function applyTheme(palette, appearance, persist) {
    const root = document.documentElement;
    if (palette && palette !== "graphite") root.setAttribute("data-palette", palette); else root.removeAttribute("data-palette");
    if (appearance && appearance !== "auto") root.setAttribute("data-theme", appearance); else root.removeAttribute("data-theme");
    if (persist) { try { localStorage.setItem(THEME_KEY, JSON.stringify({ palette, appearance })); } catch (_) { /* ignore */ } }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", cssVar("--bg"));
    syncBasemap();
    // Data colours live in CSS variables, so redraw what is on screen.
    if (state.features.length) renderFeatures(state.features, false); else renderLegend(checkedValue("color-mode") || "severity");
    if (state.chartRows.length) drawChart();
    document.querySelectorAll("#source-breakdown .bar span").forEach((bar) => { bar.style.background = cssVar(SOURCE_VARS[bar.dataset.source] || "--text-3"); });
  }
  function initTheme() {
    const saved = readTheme();
    const palette = saved.palette || "graphite", appearance = saved.appearance || "auto";
    const p = document.querySelector(`input[name="palette"][value="${palette}"]`); if (p) p.checked = true;
    const a = document.querySelector(`input[name="appearance"][value="${appearance}"]`); if (a) a.checked = true;
    el("theme-popover").addEventListener("change", () => applyTheme(checkedValue("palette"), checkedValue("appearance"), true));
    darkQuery.addEventListener("change", () => { if (checkedValue("appearance") === "auto") applyTheme(checkedValue("palette"), "auto", false); });

    const button = el("theme-button"), pop = el("theme-popover");
    const close = () => { if (pop.hidden) return; tween({ targets: pop, opacity: 0, scale: 0.96, duration: 140, easing: "easeInCubic", complete: () => { pop.hidden = true; } }); button.setAttribute("aria-expanded", "false"); };
    const open = () => { pop.hidden = false; refreshSegments(); tween({ targets: pop, opacity: [0, 1], scale: [0.96, 1], translateY: [-4, 0], duration: 220, easing: "easeOutCubic" }); button.setAttribute("aria-expanded", "true"); };
    button.addEventListener("click", () => (pop.hidden ? open() : close()));
    document.addEventListener("click", (e) => { if (!el("theme-menu").contains(e.target)) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  }

  // ------------------------------------------------------------------ map
  const map = L.map("map", { preferCanvas: true, worldCopyJump: true, zoomControl: true }).setView([22, 12], 2);
  const credit = " · Fire data: NASA FIRMS · Weather: NASA POWER";
  const baseLayers = {
    "Light": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, attribution: "Tiles © Esri, HERE, Garmin, FAO, NOAA, USGS" + credit }),
    "Dark": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}", { maxZoom: 16, attribution: "Tiles © Esri, HERE, Garmin, FAO, NOAA, USGS" + credit }),
    "Satellite": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 18, attribution: "Tiles © Esri, Maxar, Earthstar Geographics" + credit }),
    "Streets": L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' + credit }),
  };
  L.control.layers(baseLayers, null, { position: "topright", collapsed: true }).addTo(map);
  map.on("baselayerchange", (e) => { state.baseName = e.name; state.baseManual = true; });
  function syncBasemap() {
    if (state.baseManual) return; // the viewer picked a basemap by hand; leave it
    const wanted = isDark() ? "Dark" : "Light";
    if (state.baseName === wanted) return;
    Object.values(baseLayers).forEach((layer) => { if (map.hasLayer(layer)) map.removeLayer(layer); });
    baseLayers[wanted].addTo(map);
    state.baseName = wanted;
    state.baseManual = false;
  }
  const canvasRenderer = L.canvas({ padding: 0.5 });
  map.on("click", (event) => { setPin(event.latlng.lat, event.latlng.lng, true); el("predict-disclosure").open = true; });

  function setPin(lat, lon, ripple) {
    el("p-lat").value = lat.toFixed(4);
    el("p-lon").value = lon.toFixed(4);
    if (!state.pin) {
      const icon = L.divIcon({ className: "", iconSize: [20, 20], iconAnchor: [10, 10], html: '<div class="pin"><span class="ring"></span><span class="ring"></span><span class="pin-core"></span></div>' });
      state.pin = L.marker([lat, lon], { icon, draggable: true, title: "Detection to score. Drag to move.", zIndexOffset: 1000 }).addTo(map);
      state.pin.on("dragend", () => { const p = state.pin.getLatLng(); el("p-lat").value = p.lat.toFixed(4); el("p-lon").value = p.lng.toFixed(4); });
    } else {
      state.pin.setLatLng([lat, lon]);
    }
    if (ripple) pulsePin(2);
  }
  function pulsePin(loops) {
    const node = state.pin && state.pin.getElement();
    if (!node || !MOTION) return;
    const rings = node.querySelectorAll(".ring");
    anime.remove(rings);
    anime({ targets: rings, scale: [0.6, 2.6], opacity: [0.55, 0], duration: 1100, delay: anime.stagger(260), easing: "easeOutCubic", loop: loops });
  }

  // -------------------------------------------------------------- network
  async function getJSON(path, params) {
    const url = new URL(path, window.location.origin);
    Object.entries(params || {}).forEach(([k, v]) => {
      if (v === undefined || v === null || v === "") return;
      if (Array.isArray(v)) v.forEach((item) => url.searchParams.append(k, item)); else url.searchParams.set(k, v);
    });
    const response = await fetch(url);
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json();
  }
  async function postJSON(path, body) {
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(await errorMessage(response));
    return response.json();
  }
  async function errorMessage(response) {
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") return payload.detail;
      if (Array.isArray(payload.detail)) return payload.detail.map((d) => `${d.loc.slice(-1)[0]}: ${d.msg}`).join("; ");
    } catch (_) { /* ignore */ }
    return `The server answered ${response.status}.`;
  }

  let toastTimer = null;
  function toast(message) {
    const node = el("toast");
    node.textContent = message;
    node.hidden = false;
    tween({ targets: node, opacity: [0, 1], translateY: [16, 0], translateX: "-50%", duration: 320, easing: "easeOutCubic" });
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => tween({ targets: node, opacity: 0, translateY: 10, translateX: "-50%", duration: 220, easing: "easeInCubic", complete: () => { node.hidden = true; } }), 5000);
  }
  function setBusy(buttonId, busy, busyLabel, idleLabel) {
    const button = el(buttonId);
    button.disabled = busy;
    button.querySelector(".btn-label").textContent = busy ? busyLabel : idleLabel;
  }
  function setLoading(flag) {
    state.loading = flag;
    el("nav-progress").classList.toggle("on", flag);
    setBusy("apply-btn", flag, "Loading fires", "Show fires on the map");
  }
  const addDays = (iso, days) => { const d = new Date(iso + "T00:00:00Z"); d.setUTCDate(d.getUTCDate() + days); return d.toISOString().slice(0, 10); };
  const checkedValue = (name) => { const n = document.querySelector(`input[name="${name}"]:checked`); return n ? n.value : ""; };

  // -------------------------------------------------------------- filters
  function readFilters() {
    const params = {
      start_date: el("start-date").value, end_date: el("end-date").value,
      source: Array.from(document.querySelectorAll('input[name="source"]:checked')).map((i) => i.value),
      min_confidence: el("min-confidence").value, min_frp: el("min-frp").value, daynight: checkedValue("daynight"),
    };
    if (el("use-viewport").checked) {
      const b = map.getBounds(), clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
      params.bbox = [clamp(b.getWest(), -180, 180), clamp(b.getSouth(), -90, 90), clamp(b.getEast(), -180, 180), clamp(b.getNorth(), -90, 90)].map((v) => v.toFixed(4)).join(",");
    }
    return params;
  }
  function resetFilters() {
    const end = (state.health && state.health.fire_date_range.end_date) || new Date().toISOString().slice(0, 10);
    el("end-date").value = end;
    el("start-date").value = addDays(end, -6);
    document.querySelectorAll('input[name="source"]').forEach((i) => { i.checked = true; });
    el("min-confidence").value = ""; el("min-frp").value = "";
    document.querySelector('input[name="daynight"][value=""]').checked = true;
    el("limit").value = "5000";
    el("use-viewport").checked = false;
    refreshSegments();
  }

  // --------------------------------------------------------------- colours
  function hexToRgb(hex) { const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
  function palette() {
    return { high: cssVar("--severity-high"), normal: cssVar("--severity-normal"), text3: cssVar("--text-3"), track: cssVar("--surface-2"), sources: Object.fromEntries(Object.entries(SOURCE_VARS).map(([k, v]) => [k, cssVar(v)])) };
  }
  const probabilityColor = (p, pal) => mix(hexToRgb(pal.normal), hexToRgb(pal.high), Math.max(0, Math.min(1, p)));
  const frpColor = (frp, pal) => mix(hexToRgb(pal.track), hexToRgb(pal.high), Math.max(0, Math.min(1, Math.log10(1 + (frp || 0)) / 3)));
  function featureColor(props, mode, pal) {
    if (mode === "probability") return probabilityColor(props.high_frp_probability ?? 0, pal);
    if (mode === "frp") return frpColor(props.frp, pal);
    if (mode === "source") return pal.sources[props.satellite_source] || pal.text3;
    return props.is_high_frp ? pal.high : pal.normal;
  }
  function renderLegend(mode) {
    const pal = palette(), node = el("legend");
    if (mode === "probability") node.innerHTML = `<h4>Chance of a severe fire</h4><div class="ramp" style="background:linear-gradient(90deg,${probabilityColor(0, pal)},${probabilityColor(1, pal)})"></div><div class="ramp-labels"><span>0</span><span>0.5</span><span>1</span></div>`;
    else if (mode === "frp") node.innerHTML = `<h4>Measured FRP, MW</h4><div class="ramp" style="background:linear-gradient(90deg,${frpColor(0, pal)},${frpColor(30, pal)},${frpColor(1000, pal)})"></div><div class="ramp-labels"><span>0</span><span>30</span><span>1,000+</span></div>`;
    else if (mode === "source") node.innerHTML = "<h4>Satellite</h4>" + Object.keys(SOURCE_LABELS).map((k) => `<div class="legend-row"><span class="swatch" style="background:${pal.sources[k]}"></span>${SOURCE_LABELS[k]}</div>`).join("");
    else node.innerHTML = `<h4>Predicted severity</h4><div class="legend-row"><span class="swatch" style="background:${pal.high}"></span>Severe, FRP at or above threshold</div><div class="legend-row"><span class="swatch" style="background:${pal.normal}"></span>Normal</div>`;
  }

  // ------------------------------------------------------------- rendering
  const SENSOR_PIXELS = { viirs: { scan: 0.38, track: 0.36 }, modis: { scan: 1.0, track: 1.0 } };
  function popupHtml(props, lat, lon, index) {
    const tag = props.is_high_frp ? '<span class="pop-tag high">Predicted severe</span>' : '<span class="pop-tag normal">Predicted normal</span>';
    const time = props.acq_time != null ? String(props.acq_time).padStart(4, "0").replace(/(\d\d)(\d\d)/, "$1:$2") + " UTC" : "";
    const rows = [
      ["Measured FRP", props.frp != null ? `${props.frp} MW` : "Not recorded"],
      ["Chance of severe", props.high_frp_probability != null ? `${(props.high_frp_probability * 100).toFixed(1)}%` : "Not scored"],
      ["Brightness", props.brightness != null ? `${props.brightness} K` : "Not recorded"],
      ["Confidence", props.confidence != null ? `${props.confidence}%` : "Not recorded"],
      ["Satellite", `${SOURCE_LABELS[props.satellite_source] || props.satellite_source || "Unknown"}, ${props.daynight === "N" ? "night" : "day"} pass`],
      ["Weather", props.weather_matched ? "Matched to a station" : "No station nearby, values imputed"],
    ];
    const action = index != null ? `<button type="button" class="pop-action" data-score="${index}">Score this detection with its own values</button>` : "";
    return `<p class="pop-title">${props.acq_date ? dateLabel(props.acq_date) : "Detection"} ${time}</p><p class="pop-sub">${lat.toFixed(4)}, ${lon.toFixed(4)}</p>${tag}<table>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>${action}`;
  }
  // Fill the scoring form from a mapped detection so the score matches the dot.
  function fillFormFromFeature(feature) {
    const p = feature.properties, [lon, lat] = feature.geometry.coordinates;
    setPin(lat, lon, false);
    if (p.acq_date) el("p-date").value = p.acq_date;
    if (p.acq_time != null) el("p-time").value = p.acq_time;
    if (p.brightness != null) el("p-brightness").value = p.brightness;
    if (p.confidence != null) el("p-confidence").value = p.confidence;
    if (p.type != null) el("p-type").value = String(p.type);
    if (p.scan != null) el("p-scan").value = p.scan;
    if (p.track != null) el("p-track").value = p.track;
    const sensor = p.satellite_source === "modis" ? "modis" : "viirs";
    document.querySelector(`input[name="sensor"][value="${sensor}"]`).checked = true;
    el("predict-disclosure").open = true;
    refreshSegments();
  }
  map.on("popupopen", (e) => {
    const button = e.popup.getElement() && e.popup.getElement().querySelector("[data-score]");
    if (!button) return;
    button.addEventListener("click", () => {
      const feature = state.features[Number(button.dataset.score)];
      if (!feature) return;
      map.closePopup();
      fillFormFromFeature(feature);
      el("predict-form").requestSubmit();
    });
  });
  function renderFeatures(features, reveal) {
    const mode = checkedValue("color-mode") || "severity", pal = palette();
    if (state.layer) { map.removeLayer(state.layer); state.layer = null; }
    const radius = features.length > 20000 ? 2.5 : features.length > 5000 ? 3.5 : 4.5;
    state.layer = L.geoJSON({ type: "FeatureCollection", features }, {
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, { renderer: canvasRenderer, radius, color: "rgba(255,255,255,0.6)", weight: 0.6, fillColor: featureColor(feature.properties, mode, pal), fillOpacity: 0.85 }),
      onEachFeature: (feature, layer) => layer.bindPopup(() => { const [lon, lat] = feature.geometry.coordinates; return popupHtml(feature.properties, lat, lon, features.indexOf(feature)); }, { maxWidth: 300 }),
    }).addTo(map);
    renderLegend(mode);
    if (reveal !== false) tween({ targets: map.getPane("overlayPane"), opacity: [0, 1], duration: 420, easing: "easeOutCubic" });
  }
  function renderSummary(geojson, stats) {
    const meta = geojson.metadata, f = state.lastFilters;
    el("hero-range").textContent = f ? `${dateLabel(f.start_date)} to ${dateLabel(f.end_date)}${f.bbox ? ", area on screen" : ""}` : "";
    countTo("stat-total", stats ? stats.total : 0, (v) => fmt.format(Math.round(v)));
    el("hero-note").textContent = meta.truncated ? `· ${fmt.format(meta.count)} most recent drawn` : (meta.count && stats && meta.count < stats.total ? `· ${fmt.format(meta.count)} drawn` : "");
    countTo("stat-predicted", meta.predicted_high_frp != null && meta.count ? (meta.predicted_high_frp / meta.count) * 100 : 0, (v) => v.toFixed(1) + "%");
    countTo("stat-observed", stats && stats.total ? stats.high_frp_share * 100 : 0, (v) => v.toFixed(1) + "%");
    if (stats) el("thr").textContent = stats.high_frp_threshold_mw;
    el("empty-note").hidden = !!(stats && stats.total);
    const note = el("load-note");
    note.hidden = !meta.truncated;
    if (meta.truncated) note.textContent = `Only the ${fmt.format(meta.count)} most recent detections are drawn. Narrow the dates, switch on “Only the area on screen”, or raise the point limit under More options.`;

    const pal = palette(), breakdown = el("source-breakdown");
    if (stats && stats.by_source.length) {
      const max = Math.max(...stats.by_source.map((s) => s.count));
      breakdown.innerHTML = stats.by_source.map((s) => `<li><span>${SOURCE_LABELS[s.source] || s.source}</span><span class="bar"><span data-source="${s.source}" style="background:${pal.sources[s.source] || pal.text3}" data-w="${s.count / max}"></span></span><span class="count">${fmt.format(s.count)}</span></li>`).join("");
      breakdown.querySelectorAll(".bar span").forEach((bar, i) => tween({ targets: bar, scaleX: [0, Number(bar.dataset.w)], duration: 600, delay: 120 + i * 60, easing: "easeOutCubic" }));
    } else {
      breakdown.innerHTML = "";
    }
    state.chartRows = stats ? stats.by_date : [];
    const progress = { value: 0 };
    tween({ targets: progress, value: 1, duration: 650, easing: "easeOutCubic", update: () => { state.chartProgress = progress.value; drawChart(); }, complete: () => { state.chartProgress = 1; drawChart(); } });
  }
  function drawChart() {
    const rows = state.chartRows, canvas = el("daily-chart"), pal = palette();
    const dpr = window.devicePixelRatio || 1, width = canvas.clientWidth || 320, height = 96;
    canvas.width = width * dpr; canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);
    if (!rows.length) { ctx.fillStyle = pal.text3; ctx.font = "12px Manrope, system-ui"; ctx.textAlign = "center"; ctx.fillText("Nothing to chart yet", width / 2, height / 2); return; }
    const top = 6, plotH = height - top, max = Math.max(...rows.map((r) => r.count)) || 1;
    const slot = width / rows.length, gap = rows.length > 90 ? 0 : Math.min(3, slot * 0.25), barW = Math.max(1, slot - gap);
    rows.forEach((row, i) => {
      const local = Math.max(0, Math.min(1, (state.chartProgress - (i / rows.length) * 0.35) / 0.65));
      const x = i * slot, h = (row.count / max) * plotH * local, hh = (row.high_frp_count / max) * plotH * local;
      ctx.fillStyle = pal.track; ctx.fillRect(x, top, barW, plotH);
      ctx.fillStyle = pal.normal; ctx.fillRect(x, top + plotH - h, barW, h);
      ctx.fillStyle = pal.high; ctx.fillRect(x, top + plotH - hh, barW, hh);
    });
    canvas.onmousemove = (event) => {
      const rect = canvas.getBoundingClientRect(), row = rows[Math.floor((event.clientX - rect.left) / slot)];
      el("chart-caption").textContent = row ? `${dateLabel(row.date)}: ${fmt.format(row.count)} detections, ${fmt.format(row.high_frp_count)} severe.` : DEFAULT_CAPTION;
    };
    canvas.onmouseleave = () => { el("chart-caption").textContent = DEFAULT_CAPTION; };
  }

  // ----------------------------------------------------------------- loads
  async function loadFires() {
    if (state.loading) return;
    const filters = readFilters();
    if (!filters.start_date || !filters.end_date) { toast("Choose a start and an end date first."); return; }
    if (filters.start_date > filters.end_date) { toast("The start date is after the end date."); return; }
    if (!filters.source.length) { toast("Pick at least one satellite."); return; }
    setLoading(true);
    try {
      const [geojson, stats] = await Promise.all([
        getJSON("/api/v1/fires/geojson", { ...filters, limit: el("limit").value, predict: true }),
        getJSON("/api/v1/fires/stats", filters),
      ]);
      state.lastFilters = filters; state.features = geojson.features; state.stats = stats;
      if (!state.dateTouched) el("p-date").value = filters.end_date;
      renderFeatures(geojson.features, true);
      renderSummary(geojson, stats);
      if (geojson.features.length && !el("use-viewport").checked) map.flyToBounds(state.layer.getBounds().pad(0.08), { maxZoom: 8, duration: MOTION ? 0.9 : 0 });
    } catch (error) {
      toast(`The fires could not be loaded. ${error.message}`);
    } finally {
      setLoading(false);
    }
  }
  const FEATURE_LABELS = { latitude: "Latitude", longitude: "Longitude", brightness: "Brightness", confidence: "Confidence", scan: "Scan (pixel)", track: "Track (pixel)", type: "Type", acq_time: "Time of day", day_of_year: "Day of year", rh2m: "Humidity", prectotcorr: "Precipitation", gwettop: "Soil wetness" };
  async function loadModelNotes() {
    try {
      const info = await getJSON("/api/v1/model");
      state.model = info;
      const notes = el("model-notes");
      if (!info.loaded) { notes.hidden = true; return; }
      const imp = Object.entries(info.feature_importances || {}).slice(0, 6);
      const max = imp.length ? imp[0][1] : 1;
      el("importance-list").innerHTML = imp.map(([k, v]) => `<li><span>${FEATURE_LABELS[k] || k}</span><span class="bar"><span data-w="${v / max}"></span></span><span class="pct">${(v * 100).toFixed(0)}%</span></li>`).join("");
      const ext = info.training_extent;
      const rows = ext ? fmt.format(ext.rows) : null;
      el("model-extent").textContent = ext
        ? `Trained on ${rows} detections from ${dateLabel(ext.start_date)} to ${dateLabel(ext.end_date)}, latitudes ${ext.latitude_min.toFixed(1)} to ${ext.latitude_max.toFixed(1)} and longitudes ${ext.longitude_min.toFixed(1)} to ${ext.longitude_max.toFixed(1)}${ext.rows_with_weather != null ? `, ${fmt.format(ext.rows_with_weather)} with matched weather` : ""}. Scores outside that range are extrapolations.`
        : "This model artifact does not record where or when it was trained, so scores far from the weather stations may be extrapolations.";
      notes.hidden = false;
      notes.querySelectorAll(".bar span").forEach((bar, i) => tween({ targets: bar, scaleX: [0, Number(bar.dataset.w)], duration: 500, delay: 80 * i, easing: "easeOutCubic" }));
    } catch (_) { /* notes are optional */ }
  }

  async function loadHealth() {
    const pill = el("status-pill"), text = el("status-text");
    try {
      const health = await getJSON("/health");
      state.health = health;
      const backend = health.backend === "postgres" ? "PostgreSQL" : health.backend === "csv" ? "CSV sample" : health.backend;
      text.textContent = `${backend}, model ${health.model.loaded ? "ready" : "missing"}`;
      pill.className = "status " + (health.status === "ok" ? "ok" : "warn");
      pill.title = `Fire data ${health.fire_date_range.start_date ?? "?"} to ${health.fire_date_range.end_date ?? "?"}. Weather ${health.weather.start_date ?? "none"} to ${health.weather.end_date ?? ""}.`;
      if (health.fire_date_range.end_date) {
        el("start-date").min = health.fire_date_range.start_date;
        el("end-date").max = health.fire_date_range.end_date;
        el("p-date").value = health.fire_date_range.end_date;
      }
      return health;
    } catch (error) {
      text.textContent = "API unreachable";
      pill.className = "status warn";
      toast(`The API did not answer. ${error.message}`);
      return null;
    }
  }

  // ------------------------------------------------------------ prediction
  const numberOrNull = (id) => (el(id).value === "" ? null : Number(el(id).value));
  async function runPrediction(event) {
    event.preventDefault();
    const body = {
      latitude: Number(el("p-lat").value), longitude: Number(el("p-lon").value), acq_date: el("p-date").value,
      acq_time: numberOrNull("p-time"), brightness: numberOrNull("p-brightness"), confidence: numberOrNull("p-confidence"),
      scan: numberOrNull("p-scan"), track: numberOrNull("p-track"), type: Number(el("p-type").value),
    };
    setBusy("predict-btn", true, "Scoring", "Score this detection");
    try {
      const result = await postJSON("/api/v1/predict", body);
      const box = el("predict-result"), first = box.hidden;
      box.hidden = false;
      box.classList.toggle("high", result.is_high_frp);
      el("result-label").textContent = result.is_high_frp ? "Likely a severe fire" : "Likely a normal fire";
      el("result-sub").textContent = `Severe means ${result.high_frp_threshold_mw} MW or more. Cut‑off at ${Math.round(result.decision_threshold * 100)}%.`;
      el("result-threshold").style.left = `${result.decision_threshold * 100}%`;
      const caution = el("result-caution");
      if (result.in_training_extent === false) {
        caution.textContent = "This place or date lies outside the data the model was trained on, so the score is an extrapolation and will look flat across large areas.";
        caution.hidden = false;
      } else {
        caution.hidden = true;
      }
      const w = result.weather;
      const radius = (state.health && state.health.weather && state.health.weather.max_distance_km) || 100;
      const kv = [["Weather", w.matched ? `Station ${w.distance_km} km away, same day` : `No station within ${radius} km that day, values imputed`]];
      kv.unshift(["Scored as", `${checkedValue("sensor") === "modis" ? "MODIS" : "VIIRS"} pixel ${body.scan} × ${body.track} km, ${body.brightness} K, confidence ${body.confidence}%, ${dateLabel(body.acq_date)}`]);
      if (w.matched) kv.push(["Humidity", `${w.rh2m ?? "Not recorded"} %`], ["Precipitation", `${w.prectotcorr ?? "Not recorded"} mm`], ["Soil wetness", `${w.gwettop ?? "Not recorded"}`]);
      el("result-weather").innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      if (first) tween({ targets: box, opacity: [0, 1], translateY: [8, 0], duration: 360, easing: "easeOutCubic" });
      countTo("result-prob", result.high_frp_probability * 100, (v) => v.toFixed(1));
      tween({ targets: el("result-meter"), scaleX: [state.lastProb, result.high_frp_probability], duration: 700, easing: "easeOutExpo" });
      state.lastProb = result.high_frp_probability;
      setPin(body.latitude, body.longitude, false);
      pulsePin(3);
      state.pin.bindPopup(`<p class="pop-title">${(result.high_frp_probability * 100).toFixed(1)}% chance of a severe fire</p><p class="pop-sub">${result.is_high_frp ? "Predicted severe" : "Predicted normal"}</p>`, { minWidth: 230 }).openPopup();
    } catch (error) {
      toast(`The detection could not be scored. ${error.message}`);
    } finally {
      setBusy("predict-btn", false, "Scoring", "Score this detection");
    }
  }

  // --------------------------------------------------------------- sidebar
  function toggleSidebar() {
    const sidebar = el("sidebar"), button = el("toggle-sidebar");
    const hide = !document.body.classList.contains("panel-hidden");
    document.body.classList.toggle("panel-hidden", hide);
    button.setAttribute("aria-expanded", String(!hide));
    button.textContent = hide ? "Show panel" : "Hide panel";
    const small = window.innerWidth <= 860;
    const offset = small ? "120%" : -(sidebar.offsetWidth + 32);
    tween({ targets: sidebar, translateX: small ? 0 : (hide ? offset : 0), translateY: small ? (hide ? offset : 0) : 0, duration: 380, easing: "easeOutCubic" });
    setTimeout(() => map.invalidateSize(), 400);
  }

  // ------------------------------------------------------------- assistant
  const TOOL_LABELS = {
    get_data_coverage: "Data coverage",
    get_fire_statistics: "Fire statistics",
    list_fires: "Detections",
    predict_fire_severity: "Model score",
    get_model_info: "Model details",
    search_project_docs: "Project docs",
  };
  const escapeHtml = (text) => text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  // Small, safe Markdown subset: paragraphs, bullet and numbered lists, bold, inline code.
  function renderMarkdown(text) {
    const inline = (line) => escapeHtml(line).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/`([^`]+)`/g, "<code>$1</code>");
    const out = [];
    let list = null;
    const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of text.split(/\n/)) {
      const line = raw.trimEnd();
      const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
      const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
      if (bullet || numbered) {
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) { close(); out.push(`<${kind}>`); list = kind; }
        out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
      } else if (!line.trim()) {
        close();
      } else {
        close();
        const heading = line.match(/^#{1,6}\s+(.*)$/);
        out.push(heading ? `<p><strong>${inline(heading[1])}</strong></p>` : `<p>${inline(line)}</p>`);
      }
    }
    close();
    return out.join("");
  }
  function lookupLabel(call) {
    const label = TOOL_LABELS[call.name] || call.name;
    const i = call.input || {};
    if (i.start_date && i.end_date) return `${label}, ${dateLabel(i.start_date)} to ${dateLabel(i.end_date)}${i.bbox ? ", area" : ""}`;
    if (i.query) return `${label}: “${i.query}”`;
    return label;
  }

  const ask = { history: [], busy: false, enabled: true };
  async function loadAssistantStatus() {
    try {
      const status = await getJSON("/api/v1/ask/status");
      ask.enabled = status.enabled;
      if (!status.enabled) {
        const note = el("ask-status");
        note.textContent = status.reason || "The assistant is not available.";
        note.hidden = false;
        el("ask-input").disabled = true;
        el("ask-btn").disabled = true;
        el("ask-suggestions").hidden = true;
      }
    } catch (_) { /* leave it enabled; problems surface when a question is sent */ }
  }
  function appendTurn(question) {
    const turn = document.createElement("div");
    turn.className = "turn";
    turn.innerHTML = '<p class="turn-question"></p><div class="turn-answer pending">Looking it up<span class="typing" aria-hidden="true"><span></span><span></span><span></span></span></div><div class="lookups"></div>';
    turn.querySelector(".turn-question").textContent = question;
    el("ask-thread").appendChild(turn);
    tween({ targets: turn, opacity: [0, 1], translateY: [8, 0], duration: 320, easing: "easeOutCubic" });
    if (MOTION) anime({ targets: turn.querySelectorAll(".typing span"), translateY: [0, -3, 0], duration: 900, delay: anime.stagger(120), loop: true, easing: "easeInOutSine" });
    turn.scrollIntoView({ behavior: MOTION ? "smooth" : "auto", block: "nearest" });
    return turn;
  }
  async function submitQuestion(question) {
    question = question.trim();
    if (question.length < 2 || ask.busy || !ask.enabled) return;
    ask.busy = true;
    setBusy("ask-btn", true, "Thinking", "Ask");
    el("ask-input").value = "";
    el("ask-suggestions").hidden = true;
    const turn = appendTurn(question);
    const answerNode = turn.querySelector(".turn-answer");
    try {
      const result = await postJSON("/api/v1/ask", { question, history: ask.history.slice(-10) });
      answerNode.classList.remove("pending");
      answerNode.innerHTML = renderMarkdown(result.answer || "No answer was returned.");
      turn.querySelector(".lookups").innerHTML = result.tool_calls.map((call) => `<span class="lookup${call.ok ? "" : " err"}">${escapeHtml(lookupLabel(call))}</span>`).join("");
      tween({ targets: answerNode, opacity: [0, 1], duration: 260, easing: "easeOutCubic" });
      ask.history.push({ role: "user", content: question }, { role: "assistant", content: result.answer || "" });
      el("ask-reset").hidden = false;
    } catch (error) {
      answerNode.classList.remove("pending");
      answerNode.classList.add("failed");
      answerNode.textContent = `No answer. ${error.message}`;
    } finally {
      if (MOTION) anime.remove(turn.querySelectorAll(".typing span"));
      ask.busy = false;
      setBusy("ask-btn", false, "Thinking", "Ask");
      el("ask-input").focus();
    }
  }
  el("ask-form").addEventListener("submit", (e) => { e.preventDefault(); submitQuestion(el("ask-input").value); });
  el("ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitQuestion(el("ask-input").value); } });
  document.querySelectorAll(".suggestion").forEach((button) => button.addEventListener("click", () => submitQuestion(button.textContent)));
  el("ask-reset").addEventListener("click", () => {
    ask.history = [];
    el("ask-thread").innerHTML = "";
    el("ask-reset").hidden = true;
    el("ask-suggestions").hidden = !ask.enabled;
    el("ask-input").focus();
  });

  // ------------------------------------------------------------------ init
  el("filter-form").addEventListener("submit", (e) => { e.preventDefault(); loadFires(); });
  el("reset-btn").addEventListener("click", () => { resetFilters(); loadFires(); });
  el("color-seg").addEventListener("change", () => { if (state.features.length) renderFeatures(state.features, false); else renderLegend(checkedValue("color-mode")); });
  el("predict-form").addEventListener("submit", runPrediction);
  el("sensor-seg").addEventListener("change", () => {
    const pixels = SENSOR_PIXELS[checkedValue("sensor")] || SENSOR_PIXELS.viirs;
    el("p-scan").value = pixels.scan;
    el("p-track").value = pixels.track;
  });
  el("toggle-sidebar").addEventListener("click", toggleSidebar);
  window.addEventListener("resize", () => { if (state.chartRows.length) drawChart(); });

  el("p-date").addEventListener("input", () => { state.dateTouched = true; });
  initTheme();
  applyTheme(checkedValue("palette"), checkedValue("appearance"), false);
  setPin(30.05, 31.2, false);
  enterSections();
  loadHealth().then((health) => { resetFilters(); if (health && health.status !== "degraded") loadFires(); loadModelNotes(); loadAssistantStatus(); });
})();
