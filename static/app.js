"use strict";

/* ------------------------------------------------------------------ estado */

const state = {
  detectors: [],
  presets: [],
  params: {},          // { detectorKey: { paramKey: valor } }
  selected: new Set(),
  image: null,         // { image_id, filename, width, height }
  imageEl: null,
  results: [],
  metrics: [],
  groundTruth: [],
  iouThreshold: 0.3,
  zoom: null,          // null = ajustar a largura disponivel
  viewMode: "overlay",
  showLabels: true,
  showLandmarks: false,
  gtMode: false,
  drag: null,
  running: false,
};

const el = (id) => document.getElementById(id);

const dom = {
  dropzone: el("dropzone"),
  fileInput: el("file-input"),
  imageMeta: el("image-meta"),
  presetSelect: el("preset-select"),
  presetHelp: el("preset-help"),
  applyPreset: el("btn-apply-preset"),
  detectorList: el("detector-list"),
  viewer: el("viewer"),
  emptyState: el("empty-state"),
  overlayView: el("overlay-view"),
  gridView: el("grid-view"),
  canvas: el("canvas"),
  tooltip: el("tooltip"),
  statusBar: el("status-bar"),
  scoreboard: el("scoreboard"),
  metrics: el("metrics"),
  metricsIntro: el("metrics-intro"),
  histogram: el("histogram"),
  histSelect: el("hist-select"),
  iouRow: el("iou-row"),
  iouSlider: el("iou-slider"),
  iouBadge: el("iou-badge"),
  runBtn: el("btn-run"),
  exportBtn: el("btn-export"),
  modelStatus: el("model-status"),
  zoomLabel: el("zoom-label"),
  toast: el("toast"),
};

const ctx = dom.canvas.getContext("2d");

/* ---------------------------------------------------------------- helpers */

function toast(message, kind = "") {
  dom.toast.textContent = message;
  dom.toast.className = `toast ${kind}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => dom.toast.classList.add("hidden"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch (_) { /* resposta sem corpo JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

const fmt = (value, digits = 3) =>
  value === null || value === undefined ? "\u2014" : Number(value).toFixed(digits);

/* --------------------------------------------------------------- modelos */

async function refreshModels() {
  try {
    const data = await api("/api/models");
    if (data.missing_count === 0) {
      dom.modelStatus.textContent = `${data.models.length} modelos prontos`;
      dom.modelStatus.className = "pill pill-ok";
      dom.modelStatus.onclick = null;
      dom.modelStatus.style.cursor = "default";
      return true;
    }
    if (data.download && data.download.running) {
      dom.modelStatus.textContent = "baixando modelos...";
      dom.modelStatus.className = "pill pill-warn";
      setTimeout(refreshModels, 1500);
      return false;
    }
    dom.modelStatus.textContent =
      `${data.missing_count} modelos faltando (~${data.missing_mb} MB) \u2014 clique para baixar`;
    dom.modelStatus.className = "pill pill-warn";
    dom.modelStatus.style.cursor = "pointer";
    dom.modelStatus.onclick = async () => {
      await api("/api/models/download", { method: "POST" });
      refreshModels();
    };
    return false;
  } catch (error) {
    dom.modelStatus.textContent = "erro ao consultar modelos";
    dom.modelStatus.className = "pill pill-warn";
    return false;
  }
}

/* -------------------------------------------------------------- detectores */

function buildParamControl(detector, spec) {
  const wrapper = document.createElement("div");
  const current = state.params[detector.key][spec.key];

  if (spec.kind === "bool") {
    wrapper.className = "param-check-wrap";
    const label = document.createElement("label");
    label.className = "param-check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(current);
    input.addEventListener("change", () => {
      state.params[detector.key][spec.key] = input.checked;
    });
    const text = document.createElement("span");
    text.textContent = spec.label;
    label.append(input, text);
    wrapper.appendChild(label);
    return wrapper;
  }

  wrapper.className = "param";

  const head = document.createElement("div");
  head.className = "param-head";
  const name = document.createElement("span");
  name.textContent = spec.label;
  const value = document.createElement("span");
  value.className = "param-value";
  head.append(name, value);
  wrapper.appendChild(head);

  if (spec.kind === "choice") {
    value.textContent = "";
    const select = document.createElement("select");
    (spec.choices || []).forEach((choice) => {
      const option = document.createElement("option");
      option.value = choice.value;
      option.textContent = choice.label;
      if (choice.value === current) option.selected = true;
      select.appendChild(option);
    });
    select.addEventListener("change", () => {
      state.params[detector.key][spec.key] = select.value;
    });
    wrapper.appendChild(select);
  } else {
    const input = document.createElement("input");
    input.type = "range";
    input.min = spec.minimum ?? 0;
    input.max = spec.maximum ?? 100;
    input.step = spec.step ?? (spec.kind === "int" ? 1 : 0.01);
    input.value = current;
    const render = () => {
      const parsed = spec.kind === "int" ? parseInt(input.value, 10) : parseFloat(input.value);
      state.params[detector.key][spec.key] = parsed;
      value.textContent = spec.kind === "int" ? parsed : parsed.toFixed(2);
    };
    render();
    input.addEventListener("input", render);
    wrapper.appendChild(input);
    wrapper.dataset.paramKey = spec.key;
    wrapper._sync = () => {
      input.value = state.params[detector.key][spec.key];
      render();
    };
  }

  if (spec.help) {
    const help = document.createElement("p");
    help.className = "param-help";
    help.textContent = spec.help;
    wrapper.appendChild(help);
  }
  return wrapper;
}

function renderDetectors() {
  dom.detectorList.innerHTML = "";

  state.detectors.forEach((detector) => {
    const card = document.createElement("div");
    card.className = "detector";
    if (!detector.available) card.classList.add("unavailable");
    if (state.selected.has(detector.key)) card.classList.add("checked");

    const head = document.createElement("label");
    head.className = "detector-head";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = state.selected.has(detector.key);
    checkbox.disabled = !detector.available;
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(detector.key);
      else state.selected.delete(detector.key);
      card.classList.toggle("checked", checkbox.checked);
      updateRunButton();
    });

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = detector.color;

    const name = document.createElement("span");
    name.className = "detector-name";
    name.textContent = detector.label;
    name.title = detector.label;

    const tier = document.createElement("span");
    tier.className = "tier-tag";
    tier.textContent = `T${detector.tier}`;

    const disclosure = document.createElement("button");
    disclosure.className = "disclosure";
    disclosure.type = "button";
    disclosure.textContent = "\u25be";

    head.append(checkbox, swatch, name, tier, disclosure);
    card.appendChild(head);

    const body = document.createElement("div");
    body.className = "detector-body hidden";

    const description = document.createElement("p");
    description.className = "detector-desc";
    description.textContent = detector.description;
    body.appendChild(description);

    if (!detector.available && detector.unavailable_reason) {
      const warn = document.createElement("p");
      warn.className = "detector-warn";
      warn.textContent = detector.unavailable_reason;
      body.appendChild(warn);
    }

    detector.params.forEach((spec) => body.appendChild(buildParamControl(detector, spec)));
    card.appendChild(body);

    disclosure.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const hidden = body.classList.toggle("hidden");
      disclosure.textContent = hidden ? "\u25be" : "\u25b4";
    });

    dom.detectorList.appendChild(card);
  });
}

function applyPreset(presetKey) {
  const preset = state.presets.find((p) => p.key === presetKey);
  if (!preset) return;
  state.detectors.forEach((detector) => {
    detector.params.forEach((spec) => {
      if (Object.prototype.hasOwnProperty.call(preset.params, spec.key)) {
        let value = preset.params[spec.key];
        if (spec.minimum !== null && spec.minimum !== undefined) value = Math.max(spec.minimum, value);
        if (spec.maximum !== null && spec.maximum !== undefined) value = Math.min(spec.maximum, value);
        state.params[detector.key][spec.key] = spec.kind === "int" ? Math.round(value) : value;
      }
    });
  });
  renderDetectors();
  toast(`Preset "${preset.label}" aplicado.`, "success");
}

/* ------------------------------------------------------------------ upload */

async function uploadFile(file) {
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  dom.statusBar.textContent = "enviando imagem...";
  try {
    const info = await api("/api/upload", { method: "POST", body: form });
    state.image = info;
    state.results = [];
    state.metrics = [];
    state.groundTruth = [];
    state.zoom = null;

    await loadImageElement(info.image_id);
    dom.imageMeta.textContent = `${info.filename}\n${info.width} x ${info.height} px`;
    dom.imageMeta.classList.remove("hidden");
    dom.emptyState.classList.add("hidden");
    dom.overlayView.classList.remove("hidden");
    updateRunButton();
    renderAll();
    dom.statusBar.textContent = "imagem carregada. selecione os detectores e clique em Detectar.";
  } catch (error) {
    toast(error.message, "error");
    dom.statusBar.textContent = "";
  }
}

function loadImageElement(imageId) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      state.imageEl = image;
      resolve();
    };
    image.onerror = () => reject(new Error("Falha ao carregar a imagem."));
    image.src = `/api/image/${imageId}?t=${Date.now()}`;
  });
}

/* --------------------------------------------------------------- deteccao */

function updateRunButton() {
  dom.runBtn.disabled = !state.image || state.selected.size === 0 || state.running;
  dom.exportBtn.disabled = !state.image || state.results.length === 0;
}

async function runDetection() {
  if (!state.image || state.selected.size === 0) return;
  state.running = true;
  updateRunButton();
  dom.runBtn.textContent = "detectando...";

  const order = state.detectors.map((d) => d.key).filter((k) => state.selected.has(k));
  const params = {};
  order.forEach((key) => { params[key] = state.params[key]; });

  const started = performance.now();
  try {
    const data = await api("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.image.image_id, detectors: order, params }),
    });
    state.results = data.results;
    const wall = performance.now() - started;
    const totalBoxes = state.results.reduce((sum, r) => sum + r.detections.length, 0);
    dom.statusBar.textContent =
      `${state.results.length} detectores \u00b7 ${totalBoxes} caixas \u00b7 ${wall.toFixed(0)} ms no total`;
    await refreshMetrics();
    renderAll();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    state.running = false;
    dom.runBtn.textContent = "Detectar";
    updateRunButton();
  }
}

async function refreshMetrics() {
  if (state.groundTruth.length === 0 || state.results.length === 0) {
    state.metrics = [];
    return;
  }
  try {
    const data = await api("/api/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ground_truth: state.groundTruth,
        results: state.results,
        iou_threshold: state.iouThreshold,
      }),
    });
    state.metrics = data.metrics;
  } catch (error) {
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------- desenho 2D */

function effectiveZoom() {
  if (!state.imageEl) return 1;
  if (state.zoom !== null) return state.zoom;
  const available = dom.viewer.clientWidth - 36;
  return Math.min(1, available / state.imageEl.naturalWidth);
}

function isFalseFace(detectorKey, index) {
  if (state.groundTruth.length === 0) return null;
  const result = state.results.find((r) => r.detector === detectorKey);
  if (!result) return null;
  const detection = result.detections[index];
  return !state.groundTruth.some((gt) => iou(detection, gt) >= state.iouThreshold);
}

function iou(a, b) {
  const interW = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
  const interH = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
  if (interW <= 0 || interH <= 0) return 0;
  const inter = interW * interH;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function drawScene(targetCtx, zoom, results, options = {}) {
  const image = state.imageEl;
  if (!image) return;

  targetCtx.setTransform(1, 0, 0, 1, 0, 0);
  targetCtx.clearRect(0, 0, targetCtx.canvas.width, targetCtx.canvas.height);
  targetCtx.drawImage(image, 0, 0, image.naturalWidth * zoom, image.naturalHeight * zoom);

  const labels = options.labels ?? state.showLabels;
  const landmarks = options.landmarks ?? state.showLandmarks;

  results.forEach((result) => {
    if (!result.ok) return;
    result.detections.forEach((detection, index) => {
      const fake = isFalseFace(result.detector, index);
      const x = detection.x * zoom;
      const y = detection.y * zoom;
      const w = detection.w * zoom;
      const h = detection.h * zoom;

      // Espessura proporcional a confianca reproduz a convencao do Harvey:
      // borda grossa e uma deteccao forte, borda fina e uma candidata fraca.
      const width = Math.max(1, 1 + detection.score * 3);

      targetCtx.lineWidth = width;
      targetCtx.strokeStyle = result.color;
      targetCtx.setLineDash(fake === true ? [5, 3] : []);
      targetCtx.strokeRect(x, y, w, h);
      targetCtx.setLineDash([]);

      if (fake === false) {
        targetCtx.lineWidth = 1;
        targetCtx.strokeStyle = "rgba(255,255,255,0.85)";
        targetCtx.strokeRect(x - width, y - width, w + width * 2, h + width * 2);
      }

      if (landmarks && detection.landmarks) {
        targetCtx.fillStyle = result.color;
        detection.landmarks.forEach((point) => {
          targetCtx.beginPath();
          targetCtx.arc(point[0] * zoom, point[1] * zoom, 1.8, 0, Math.PI * 2);
          targetCtx.fill();
        });
      }

      if (labels && w > 26) {
        const text = detection.score.toFixed(2);
        targetCtx.font = "10px ui-monospace, monospace";
        const textWidth = targetCtx.measureText(text).width;
        targetCtx.fillStyle = "rgba(10,10,12,0.82)";
        targetCtx.fillRect(x, Math.max(0, y - 13), textWidth + 7, 13);
        targetCtx.fillStyle = result.color;
        targetCtx.fillText(text, x + 3.5, Math.max(9.5, y - 3.5));
      }
    });
  });

  state.groundTruth.forEach((box, index) => {
    targetCtx.lineWidth = 2;
    targetCtx.strokeStyle = "#00e5ff";
    targetCtx.setLineDash([9, 5]);
    targetCtx.strokeRect(box.x * zoom, box.y * zoom, box.w * zoom, box.h * zoom);
    targetCtx.setLineDash([]);
    if (labels) {
      const text = `rosto real ${index + 1}`;
      targetCtx.font = "10px ui-monospace, monospace";
      const textWidth = targetCtx.measureText(text).width;
      const ty = box.y * zoom + box.h * zoom;
      targetCtx.fillStyle = "rgba(0,229,255,0.9)";
      targetCtx.fillRect(box.x * zoom, ty, textWidth + 7, 13);
      targetCtx.fillStyle = "#04262b";
      targetCtx.fillText(text, box.x * zoom + 3.5, ty + 9.5);
    }
  });

  if (state.drag && options.showDrag !== false) {
    const d = normalizedDrag();
    targetCtx.lineWidth = 2;
    targetCtx.strokeStyle = "#00e5ff";
    targetCtx.setLineDash([5, 4]);
    targetCtx.strokeRect(d.x * zoom, d.y * zoom, d.w * zoom, d.h * zoom);
    targetCtx.setLineDash([]);
  }
}

function renderOverlay() {
  if (!state.imageEl) return;
  const zoom = effectiveZoom();
  dom.canvas.width = Math.round(state.imageEl.naturalWidth * zoom);
  dom.canvas.height = Math.round(state.imageEl.naturalHeight * zoom);
  dom.canvas.classList.toggle("gt-mode", state.gtMode);
  drawScene(ctx, zoom, state.results);
  dom.zoomLabel.textContent = state.zoom === null ? "ajustar" : `${Math.round(zoom * 100)}%`;
}

function renderGrid() {
  dom.gridView.innerHTML = "";
  if (!state.imageEl) return;

  state.results.forEach((result) => {
    const cell = document.createElement("div");
    cell.className = "grid-cell";

    const head = document.createElement("div");
    head.className = "grid-cell-head";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = result.color;
    const name = document.createElement("span");
    name.textContent = result.label;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = result.ok ? `${result.detections.length} caixas` : "erro";
    head.append(swatch, name, count);
    cell.appendChild(head);

    const canvas = document.createElement("canvas");
    const cellWidth = 420;
    const zoom = cellWidth / state.imageEl.naturalWidth;
    canvas.width = cellWidth;
    canvas.height = Math.round(state.imageEl.naturalHeight * zoom);
    drawScene(canvas.getContext("2d"), zoom, [result], {
      labels: false,
      landmarks: false,
      showDrag: false,
    });
    cell.appendChild(canvas);

    const foot = document.createElement("div");
    foot.className = "grid-cell-foot";
    foot.textContent = result.ok
      ? `${result.elapsed_ms.toFixed(0)} ms`
      : (result.error || "falhou");
    cell.appendChild(foot);

    dom.gridView.appendChild(cell);
  });
}

/* ------------------------------------------------------------- scoreboard */

function renderScoreboard() {
  if (state.results.length === 0) {
    dom.scoreboard.innerHTML = '<p class="help">Rode uma deteccao para ver os resultados.</p>';
    return;
  }
  dom.scoreboard.innerHTML = "";
  state.results.forEach((result) => {
    const row = document.createElement("div");
    row.className = "score-row";
    if (!result.ok) row.classList.add("failed");

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = result.color;

    const name = document.createElement("span");
    name.className = "score-name";
    name.textContent = result.label;
    name.title = result.label;

    const count = document.createElement("span");
    count.className = "score-count";
    count.style.color = result.ok ? result.color : "var(--danger)";
    count.textContent = result.ok ? result.detections.length : "!";

    const time = document.createElement("span");
    time.className = "score-time";
    time.textContent = result.ok ? `${result.elapsed_ms.toFixed(0)} ms` : "";

    row.append(swatch, name, count, time);
    dom.scoreboard.appendChild(row);

    if (!result.ok && result.error) {
      const error = document.createElement("div");
      error.className = "score-error";
      error.textContent = result.error;
      dom.scoreboard.appendChild(error);
    }
  });
}

/* ---------------------------------------------------------------- metricas */

function verdict(metric) {
  // Zero caixas nao e resistencia: o detector tambem nao achou o rosto real,
  // entao ele nao esta enxergando nada nesta imagem e nao diz nada sobre o
  // padrao.
  if (metric.total === 0) {
    return { label: "nada detectado", className: "verdict-mid", compromised: false };
  }
  // O padrao "venceu" quando o sistema, olhando so para a maior caixa ou para
  // a de maior confianca, seria levado a uma face falsa.
  if (metric.top_box_is_fake || metric.largest_box_is_fake) {
    return { label: "padrao venceu", className: "verdict-bad", compromised: true };
  }
  if (metric.false_faces > 0) {
    return { label: "ruido gerado", className: "verdict-mid", compromised: false };
  }
  return { label: "detector resistiu", className: "verdict-good", compromised: false };
}

function metricItem(key, value, className = "") {
  const item = document.createElement("div");
  item.className = "metric-item";
  const k = document.createElement("span");
  k.className = "k";
  k.textContent = key;
  const v = document.createElement("span");
  v.className = `v ${className}`;
  v.textContent = value;
  item.append(k, v);
  return item;
}

function renderMetrics() {
  const hasGt = state.groundTruth.length > 0;
  dom.metricsIntro.classList.toggle("hidden", hasGt);
  dom.iouRow.classList.toggle("hidden", !hasGt);
  dom.iouBadge.textContent = `IoU ${state.iouThreshold.toFixed(2)}`;
  dom.metrics.innerHTML = "";

  if (!hasGt || state.metrics.length === 0) return;

  state.metrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "metric-card";

    const decision = verdict(metric);
    if (decision.compromised) card.classList.add("compromised");

    const head = document.createElement("div");
    head.className = "metric-head";
    const result = state.results.find((r) => r.detector === metric.detector);
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = result ? result.color : "#888";
    const name = document.createElement("span");
    name.textContent = metric.label;
    const badge = document.createElement("span");
    badge.className = `metric-verdict ${decision.className}`;
    badge.textContent = decision.label;
    head.append(swatch, name, badge);
    card.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "metric-grid";
    grid.appendChild(
      metricItem("faces falsas", metric.false_faces, metric.false_faces > 0 ? "bad" : "good")
    );
    grid.appendChild(metricItem("rostos reais achados", metric.true_positives));
    grid.appendChild(metricItem("score do rosto real", fmt(metric.true_face_score)));
    grid.appendChild(
      metricItem(
        "maior score falso",
        fmt(metric.max_false_score),
        metric.max_false_score !== null &&
          metric.true_face_score !== null &&
          metric.max_false_score > metric.true_face_score
          ? "bad"
          : ""
      )
    );

    const ratioItem = metricItem(
      "decoy_ratio (falso / real)",
      fmt(metric.decoy_ratio, 2),
      metric.decoy_ratio !== null && metric.decoy_ratio >= 1 ? "bad" : ""
    );
    ratioItem.classList.add("wide");
    grid.appendChild(ratioItem);
    card.appendChild(grid);

    const notes = [];
    if (metric.largest_box_is_fake === true) {
      notes.push("A maior caixa detectada e falsa \u2014 um sistema que usa \"a maior face\" seria enganado.");
    }
    if (metric.top_box_is_fake === true) {
      notes.push("A caixa de maior confianca e falsa \u2014 um sistema que usa \"a face mais confiante\" seria enganado.");
    }
    if (metric.true_positives === 0 && metric.total > 0) {
      notes.push("O rosto verdadeiro nao foi detectado por este algoritmo.");
    }
    if (notes.length) {
      const note = document.createElement("p");
      note.className = "metric-note";
      note.textContent = notes.join(" ");
      card.appendChild(note);
    }

    dom.metrics.appendChild(card);
  });
}

/* -------------------------------------------------------------- histograma */

function renderHistogramSelect() {
  const previous = dom.histSelect.value;
  dom.histSelect.innerHTML = "";
  const usable = state.metrics.length ? state.metrics : [];
  usable.forEach((metric) => {
    const option = document.createElement("option");
    option.value = metric.detector;
    option.textContent = metric.label;
    dom.histSelect.appendChild(option);
  });
  if (usable.some((m) => m.detector === previous)) dom.histSelect.value = previous;
  renderHistogram();
}

function renderHistogram() {
  dom.histogram.innerHTML = "";
  const metric = state.metrics.find((m) => m.detector === dom.histSelect.value);
  if (!metric || metric.score_histogram.length === 0) {
    dom.histogram.innerHTML =
      '<p class="help" style="align-self:center">Marque o rosto verdadeiro e rode a deteccao.</p>';
    return;
  }
  const peak = Math.max(
    ...metric.score_histogram.map((bin) => bin.true_count + bin.false_count),
    1
  );
  metric.score_histogram.forEach((bin) => {
    const column = document.createElement("div");
    column.className = "hist-bin";
    column.title =
      `${bin.start.toFixed(2)} a ${bin.end.toFixed(2)}\n` +
      `reais: ${bin.true_count}\nfalsas: ${bin.false_count}`;
    if (bin.false_count) {
      const seg = document.createElement("div");
      seg.className = "hist-seg false-seg";
      seg.style.height = `${(bin.false_count / peak) * 100}%`;
      column.appendChild(seg);
    }
    if (bin.true_count) {
      const seg = document.createElement("div");
      seg.className = "hist-seg true-seg";
      seg.style.height = `${(bin.true_count / peak) * 100}%`;
      column.appendChild(seg);
    }
    dom.histogram.appendChild(column);
  });
}

/* ------------------------------------------------------------------ render */

function renderAll() {
  const hasImage = Boolean(state.imageEl);
  dom.emptyState.classList.toggle("hidden", hasImage);
  dom.overlayView.classList.toggle("hidden", !hasImage || state.viewMode !== "overlay");
  dom.gridView.classList.toggle("hidden", !hasImage || state.viewMode !== "grid");

  if (hasImage && state.viewMode === "overlay") renderOverlay();
  if (hasImage && state.viewMode === "grid") renderGrid();

  renderScoreboard();
  renderMetrics();
  renderHistogramSelect();
  updateRunButton();
}

/* ------------------------------------------------------- interacao no canvas */

function canvasPoint(event) {
  const rect = dom.canvas.getBoundingClientRect();
  const zoom = effectiveZoom();
  return {
    x: (event.clientX - rect.left) / zoom,
    y: (event.clientY - rect.top) / zoom,
  };
}

function normalizedDrag() {
  const d = state.drag;
  return {
    x: Math.min(d.startX, d.currentX),
    y: Math.min(d.startY, d.currentY),
    w: Math.abs(d.currentX - d.startX),
    h: Math.abs(d.currentY - d.startY),
  };
}

dom.canvas.addEventListener("mousedown", (event) => {
  if (!state.gtMode || !state.imageEl) return;
  const point = canvasPoint(event);
  state.drag = { startX: point.x, startY: point.y, currentX: point.x, currentY: point.y };
});

dom.canvas.addEventListener("mousemove", (event) => {
  if (!state.imageEl) return;
  const point = canvasPoint(event);

  if (state.drag) {
    state.drag.currentX = point.x;
    state.drag.currentY = point.y;
    renderOverlay();
    return;
  }

  if (state.gtMode) return;

  // Tooltip: procura a menor caixa sob o cursor, para que caixas pequenas
  // dentro de caixas grandes continuem alcancaveis.
  let best = null;
  state.results.forEach((result) => {
    if (!result.ok) return;
    result.detections.forEach((detection, index) => {
      if (
        point.x >= detection.x && point.x <= detection.x + detection.w &&
        point.y >= detection.y && point.y <= detection.y + detection.h
      ) {
        const area = detection.w * detection.h;
        if (!best || area < best.area) best = { result, detection, index, area };
      }
    });
  });

  if (!best) {
    dom.tooltip.classList.add("hidden");
    return;
  }

  const { result, detection, index } = best;
  const fake = isFalseFace(result.detector, index);
  const lines = [
    result.label,
    `score   ${detection.score.toFixed(4)}`,
    `bruto   ${detection.raw_score === null ? "\u2014" : detection.raw_score.toFixed(4)}`,
  ];
  if (detection.neighbors !== null && detection.neighbors !== undefined) {
    lines.push(`vizinhos ${detection.neighbors}`);
  }
  lines.push(
    `caixa   ${Math.round(detection.x)},${Math.round(detection.y)} ` +
    `${Math.round(detection.w)}x${Math.round(detection.h)}`
  );
  if (fake !== null) lines.push(fake ? "-> FACE FALSA" : "-> rosto verdadeiro");

  dom.tooltip.textContent = lines.join("\n");
  dom.tooltip.classList.remove("hidden");
  const rect = dom.canvas.getBoundingClientRect();
  dom.tooltip.style.left = `${event.clientX - rect.left + 14}px`;
  dom.tooltip.style.top = `${event.clientY - rect.top + 14}px`;
  dom.tooltip.style.borderColor = result.color;
});

dom.canvas.addEventListener("mouseleave", () => dom.tooltip.classList.add("hidden"));

window.addEventListener("mouseup", async () => {
  if (!state.drag) return;
  const box = normalizedDrag();
  state.drag = null;
  if (box.w >= 6 && box.h >= 6) {
    state.groundTruth.push(box);
    await refreshMetrics();
  }
  renderAll();
});

/* -------------------------------------------------------------- eventos UI */

dom.dropzone.addEventListener("click", () => dom.fileInput.click());
dom.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    dom.fileInput.click();
  }
});
dom.fileInput.addEventListener("change", () => uploadFile(dom.fileInput.files[0]));

["dragenter", "dragover"].forEach((name) =>
  dom.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dom.dropzone.classList.add("dragging");
  })
);
["dragleave", "drop"].forEach((name) =>
  dom.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dom.dropzone.classList.remove("dragging");
  })
);
dom.dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

el("btn-example").addEventListener("click", async () => {
  try {
    const response = await fetch("/static/exemplos/padrao_falsas_faces.png");
    if (!response.ok) throw new Error("Imagem de exemplo nao encontrada.");
    const blob = await response.blob();
    await uploadFile(new File([blob], "padrao_falsas_faces.png", { type: "image/png" }));
  } catch (error) {
    toast(error.message, "error");
  }
});

window.addEventListener("paste", (event) => {
  const item = Array.from(event.clipboardData?.items || []).find((i) =>
    i.type.startsWith("image/")
  );
  if (item) uploadFile(item.getAsFile());
});

dom.runBtn.addEventListener("click", runDetection);

el("btn-select-all").addEventListener("click", () => {
  state.detectors.filter((d) => d.available).forEach((d) => state.selected.add(d.key));
  renderDetectors();
  updateRunButton();
});

el("btn-select-none").addEventListener("click", () => {
  state.selected.clear();
  renderDetectors();
  updateRunButton();
});

el("btn-select-fast").addEventListener("click", () => {
  // dlib CNN e o HOG com upsample sao ordens de magnitude mais lentos que o
  // resto, entao o atalho "rapidos" deixa os dois de fora.
  const slow = new Set(["dlib_cnn", "dlib_hog"]);
  state.selected.clear();
  state.detectors
    .filter((d) => d.available && !slow.has(d.key))
    .forEach((d) => state.selected.add(d.key));
  renderDetectors();
  updateRunButton();
});

dom.presetSelect.addEventListener("change", () => {
  const preset = state.presets.find((p) => p.key === dom.presetSelect.value);
  dom.presetHelp.textContent = preset ? preset.description : "";
});

dom.applyPreset.addEventListener("click", () => applyPreset(dom.presetSelect.value));

document.querySelectorAll("#view-mode .seg").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("#view-mode .seg").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    state.viewMode = button.dataset.mode;
    renderAll();
  });
});

el("chk-gt-mode").addEventListener("change", (event) => {
  state.gtMode = event.target.checked;
  dom.tooltip.classList.add("hidden");
  renderOverlay();
});

el("chk-labels").addEventListener("change", (event) => {
  state.showLabels = event.target.checked;
  renderAll();
});

el("chk-landmarks").addEventListener("change", (event) => {
  state.showLandmarks = event.target.checked;
  renderAll();
});

el("btn-clear-gt").addEventListener("click", async () => {
  state.groundTruth = [];
  state.metrics = [];
  renderAll();
});

el("btn-zoom-in").addEventListener("click", () => {
  state.zoom = Math.min(6, effectiveZoom() * 1.25);
  renderOverlay();
});

el("btn-zoom-out").addEventListener("click", () => {
  state.zoom = Math.max(0.05, effectiveZoom() / 1.25);
  renderOverlay();
});

el("btn-zoom-fit").addEventListener("click", () => {
  state.zoom = null;
  renderOverlay();
});

dom.iouSlider.addEventListener("input", async (event) => {
  state.iouThreshold = parseFloat(event.target.value);
  dom.iouBadge.textContent = `IoU ${state.iouThreshold.toFixed(2)}`;
  await refreshMetrics();
  renderAll();
});

dom.histSelect.addEventListener("change", renderHistogram);

window.addEventListener("resize", () => {
  if (state.zoom === null) renderOverlay();
});

/* ------------------------------------------------------------------ export */

dom.exportBtn.addEventListener("click", async () => {
  if (!state.image || state.results.length === 0) return;
  dom.exportBtn.disabled = true;
  try {
    // Exporta na resolucao original, independente do zoom da tela.
    const offscreen = document.createElement("canvas");
    offscreen.width = state.imageEl.naturalWidth;
    offscreen.height = state.imageEl.naturalHeight;
    drawScene(offscreen.getContext("2d"), 1, state.results, { showDrag: false });

    const label = (state.image.filename || "run").replace(/\.[^.]+$/, "");
    const data = await api("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_id: state.image.image_id,
        label,
        image: state.image,
        ground_truth: state.groundTruth,
        iou_threshold: state.iouThreshold,
        results: state.results,
        metrics: state.metrics,
        annotated_png: offscreen.toDataURL("image/png"),
      }),
    });
    toast(`Exportado em ${data.folder}`, "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    dom.exportBtn.disabled = false;
  }
});

/* ------------------------------------------------------------------- boot */

async function boot() {
  refreshModels();
  try {
    const data = await api("/api/detectors");
    state.detectors = data.detectors;
    state.presets = data.presets;

    state.detectors.forEach((detector) => {
      state.params[detector.key] = {};
      detector.params.forEach((spec) => {
        state.params[detector.key][spec.key] = spec.default;
      });
      if (detector.available) state.selected.add(detector.key);
    });

    state.presets.forEach((preset) => {
      const option = document.createElement("option");
      option.value = preset.key;
      option.textContent = preset.label;
      dom.presetSelect.appendChild(option);
    });
    if (state.presets.length) {
      dom.presetSelect.value = state.presets[0].key;
      dom.presetHelp.textContent = state.presets[0].description;
      applyPresetSilently(state.presets[0].key);
    }

    renderDetectors();
    updateRunButton();

    const unavailable = state.detectors.filter((d) => !d.available);
    if (unavailable.length) {
      dom.statusBar.textContent =
        `${unavailable.length} detector(es) indisponivel(is). Abra o detector na lista para ver o motivo.`;
    }
  } catch (error) {
    toast(`Falha ao carregar os detectores: ${error.message}`, "error");
  }
}

function applyPresetSilently(presetKey) {
  const preset = state.presets.find((p) => p.key === presetKey);
  if (!preset) return;
  state.detectors.forEach((detector) => {
    detector.params.forEach((spec) => {
      if (Object.prototype.hasOwnProperty.call(preset.params, spec.key)) {
        let value = preset.params[spec.key];
        if (spec.minimum !== null && spec.minimum !== undefined) value = Math.max(spec.minimum, value);
        if (spec.maximum !== null && spec.maximum !== undefined) value = Math.min(spec.maximum, value);
        state.params[detector.key][spec.key] = spec.kind === "int" ? Math.round(value) : value;
      }
    });
  });
}

boot();
