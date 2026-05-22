const API_BASE =
  window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.origin}`;

const imageInput = document.getElementById("imageInput");
const detectBtn = document.getElementById("detectBtn");
const previewImage = document.getElementById("previewImage");
const annotatedImage = document.getElementById("annotatedImage");
const detectionsHeading = document.getElementById("detectionsHeading");

const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");
const detectionList = document.getElementById("detectionList");

const inferenceTime = document.getElementById("inferenceTime");
const modelSize = document.getElementById("modelSize");
const runtimeDevice = document.getElementById("runtimeDevice");
const baselineMap = document.getElementById("baselineMap");
const statusText = document.getElementById("statusText");

const navLinks = document.querySelectorAll(".nav-link");
const pages = document.querySelectorAll(".page");

const latestInference = document.getElementById("latestInference");
const latestRuntime = document.getElementById("latestRuntime");
const latestDetections = document.getElementById("latestDetections");
const leaderboardOverall = document.getElementById("leaderboardOverall");
const leaderboardFlops = document.getElementById("leaderboardFlops");
const leaderboardLatency = document.getElementById("leaderboardLatency");
const leaderboardPlate = document.getElementById("leaderboardPlate");
const leaderboardVehicle = document.getElementById("leaderboardVehicle");
const leaderboardOcr = document.getElementById("leaderboardOcr");
const leaderboardRobustness = document.getElementById("leaderboardRobustness");
const leaderboardTimestamp = document.getElementById("leaderboardTimestamp");

const modeButtons = document.querySelectorAll(".mode-btn");
const ocrConfidenceWrap = document.getElementById("ocrConfidenceControl");
const ocrConfidenceSlider = document.getElementById("ocrConfidenceSlider");
const ocrConfidenceValue = document.getElementById("ocrConfidenceValue");
const anprTimingsCard = document.getElementById("anprTimingsCard");
const anprDetectionTime = document.getElementById("anprDetectionTime");
const anprOcrTime = document.getElementById("anprOcrTime");
const anprTotalTime = document.getElementById("anprTotalTime");
const ocrBackend = document.getElementById("ocrBackend");

const setFrameState = (img, hasImage) => {
  const frame = img?.parentElement;
  if (!frame) return;
  frame.classList.toggle("has-image", Boolean(hasImage));
};

const setImageSrc = (img, src) => {
  if (!img) return;
  if (src) {
    img.src = src;
    setFrameState(img, true);
  } else {
    img.removeAttribute("src");
    setFrameState(img, false);
  }
};

let selectedMode = "vehicle";

const modeUi = {
  vehicle: {
    endpoint: "/detect/vehicle",
    buttonText: "Detect Vehicles",
    heading: "Detected Vehicles",
    readyText: "Image ready. Click Detect Vehicles.",
    emptyText: "No vehicles detected in this image.",
    doneSuffix: "vehicle(s) detected.",
  },
  plate: {
    endpoint: "/detect/plate",
    buttonText: "Detect Plates",
    heading: "Detected Plates",
    readyText: "Image ready. Click Detect Plates.",
    emptyText: "No plates detected in this image.",
    doneSuffix: "plate(s) detected.",
  },
  anpr: {
    endpoint: "/anpr",
    buttonText: "Run ANPR",
    heading: "ANPR Results",
    readyText: "Image ready. Click Run ANPR.",
    emptyText: "No vehicle/plate detections found for OCR in this image.",
    doneSuffix: "vehicle-linked plate result(s) processed.",
  },
};

const setStatus = (msg) => {
  statusText.textContent = msg;
};

const asCheckText = (value, pass) => {
  const state = pass ? "PASS" : "FAIL";
  return `${value} (${state})`;
};

const loadLeaderboardMetrics = async () => {
  if (!leaderboardOverall) {
    return;
  }

  try {
    const response = await fetch(`/runs/metrics/leaderboard_metrics.json?t=${Date.now()}`);
    if (!response.ok) {
      leaderboardOverall.textContent = "Report not generated";
      return;
    }

    const payload = await response.json();
    const metrics = payload.metrics || {};
    const checks = payload.checks || {};

    const flops = Number(metrics.efficiency?.flops_g ?? 0).toFixed(3);
    const latency = Number(metrics.efficiency?.latency_ms?.mean ?? 0).toFixed(2);
    const plateMap = Number(metrics.plate_detection?.map50 ?? 0).toFixed(4);
    const vehicleMap = Number(metrics.vehicle_detection?.map50 ?? 0).toFixed(4);
    const ocr = Number(metrics.ocr?.character_accuracy ?? 0).toFixed(4);
    const robust = Number(metrics.robustness?.overall_retention ?? 0).toFixed(4);

    leaderboardFlops.textContent = asCheckText(flops, Boolean(checks.flops_leq_5));
    leaderboardLatency.textContent = asCheckText(`${latency} ms`, Boolean(checks.latency_leq_250ms));
    leaderboardPlate.textContent = asCheckText(plateMap, Boolean(checks.plate_map50_geq_0_85));
    leaderboardVehicle.textContent = asCheckText(vehicleMap, Boolean(checks.vehicle_map50_geq_0_50));
    leaderboardOcr.textContent = asCheckText(ocr, Boolean(checks.ocr_char_acc_geq_0_80));
    leaderboardRobustness.textContent = asCheckText(robust, Boolean(checks.robustness_retention_geq_0_70));

    leaderboardOverall.textContent = checks.all_thresholds_met
      ? "Qualified"
      : "Not qualified yet";
    leaderboardTimestamp.textContent = metrics.timestamp || "-";
  } catch (error) {
    leaderboardOverall.textContent = "Report not generated";
  }
};

const setActivePage = (pageId) => {
  pages.forEach((page) => {
    page.classList.toggle("active", page.id === pageId);
  });
  navLinks.forEach((link) => {
    link.classList.toggle("active", link.dataset.page === pageId);
  });
  window.location.hash = pageId;
};

navLinks.forEach((link) => {
  link.addEventListener("click", () => setActivePage(link.dataset.page));
});

const initialPageFromHash = window.location.hash.replace("#", "");
if (initialPageFromHash && document.getElementById(initialPageFromHash)) {
  setActivePage(initialPageFromHash);
}

const updateConfidenceLabel = () => {
  confidenceValue.textContent = Number(confidenceSlider.value).toFixed(2);
};

const updateOcrConfidenceLabel = () => {
  ocrConfidenceValue.textContent = Number(ocrConfidenceSlider.value).toFixed(2);
};

updateConfidenceLabel();
updateOcrConfidenceLabel();
confidenceSlider.addEventListener("input", updateConfidenceLabel);
ocrConfidenceSlider.addEventListener("input", updateOcrConfidenceLabel);

const setDetectionMode = (mode) => {
  selectedMode = mode;
  modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));

  detectBtn.textContent = modeUi[mode].buttonText;
  detectionsHeading.textContent = modeUi[mode].heading;
  ocrConfidenceWrap.style.display = mode === "anpr" ? "grid" : "none";
  anprTimingsCard.style.display = mode === "anpr" ? "block" : "none";
  setStatus(modeUi[mode].readyText);
  detectionList.innerHTML = "";
};

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => setDetectionMode(btn.dataset.mode));
});

setDetectionMode("vehicle");
loadLeaderboardMetrics();

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  setImageSrc(previewImage, url);
  setImageSrc(annotatedImage, "");
  detectionList.innerHTML = "";

  inferenceTime.textContent = "-";
  modelSize.textContent = "-";
  runtimeDevice.textContent = "-";
  baselineMap.textContent = "";
  anprDetectionTime.textContent = "-";
  anprOcrTime.textContent = "-";
  anprTotalTime.textContent = "-";
  ocrBackend.textContent = "-";

  setStatus(modeUi[selectedMode].readyText);
});

previewImage.addEventListener("error", () => setFrameState(previewImage, false));
annotatedImage.addEventListener("error", () => setFrameState(annotatedImage, false));

const renderClassicDetections = (detections) => {
  if (!detections || detections.length === 0) {
    detectionList.innerHTML = `<li>${modeUi[selectedMode].emptyText}</li>`;
    return;
  }

  detections.forEach((det, index) => {
    const item = document.createElement("li");
    const confidencePct = Math.round(det.confidence * 100);
    const bbox = det.bbox.join(", ");
    item.innerHTML = `
      <div class="det-row">
        <span class="det-title">${index + 1}. ${det.label}</span>
        <span class="det-conf">${confidencePct}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width: ${confidencePct}%"></div>
      </div>
      <div class="det-bbox">bbox=[${bbox}]</div>
    `;
    detectionList.appendChild(item);
  });
};

const renderAnprDetections = (detections) => {
  if (!detections || detections.length === 0) {
    detectionList.innerHTML = `<li>${modeUi[selectedMode].emptyText}</li>`;
    return;
  }

  detections.forEach((det, index) => {
    const item = document.createElement("li");
    const plateText = det.cleaned_text || det.plate_text || "(unreadable)";
    const rawText = det.raw_text || "-";
    const method = det.ocr_method || det.method_used || "-";
    const variant = det.source_variant || "-";
    const detPct = Math.round((det.det_confidence || 0) * 100);
    const ocrPct = Math.round(((det.ocr_confidence ?? det.confidence) || 0) * 100);
    const plateBbox = (det.plate_bbox || det.bbox || []).join(", ");
    const vehicleBbox = (det.vehicle_bbox || []).join(", ");
    const perItemTime = det.inference_time || "-";
    const debugWeb = det.debug?.web || {};
    const originalUrl = debugWeb.original || "";
    const cropUrl = debugWeb.crop || "";
    const variantUrls = debugWeb.variants || {};
    const variantOrder = ["original", "clahe", "otsu", "sharpen_otsu"];
    const thumbs = variantOrder
      .map((name) =>
        variantUrls[name]
          ? `<figure class="debug-thumb"><img src="${variantUrls[name]}" alt="${name}" /><figcaption>${name}</figcaption></figure>`
          : ""
      )
      .join("");
    const originalBlock = originalUrl
      ? `<figure class="debug-thumb"><img src="${originalUrl}" alt="original" /><figcaption>original</figcaption></figure>`
      : "";
    const cropBlock = cropUrl
      ? `<figure class="debug-thumb"><img src="${cropUrl}" alt="crop" /><figcaption>crop</figcaption></figure>`
      : "";
    item.innerHTML = `
      <div class="det-row">
        <span class="det-title">${index + 1}. ${plateText}</span>
        <span class="det-conf">DET ${detPct}% | OCR ${ocrPct}%</span>
      </div>
      <div class="det-bbox">raw=${rawText} | method=${method} | variant=${variant} | time=${perItemTime}</div>
      <div class="det-bbox">vehicle_bbox=[${vehicleBbox || "-"}] | plate_bbox=[${plateBbox || "-"}]</div>
      <div class="det-bbox">indian_format_like=${det.indian_format_like}</div>
      <details class="debug-panel">
        <summary>Debug Plate Views</summary>
        <div class="debug-grid">
          ${originalBlock}
          ${cropBlock}
          ${thumbs}
        </div>
      </details>
    `;
    detectionList.appendChild(item);
  });
};

detectBtn.addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus("Please upload an image first.");
    return;
  }

  const mode = modeUi[selectedMode];
  const endpoint = `${API_BASE}${mode.endpoint}`;

  const formData = new FormData();
  formData.append("file", file);

  if (selectedMode === "anpr") {
    formData.append("det_confidence", confidenceSlider.value);
    formData.append("ocr_confidence", ocrConfidenceSlider.value);
  } else {
    formData.append("confidence", confidenceSlider.value);
  }

  detectBtn.disabled = true;
  detectionList.innerHTML = "";
  setStatus(`Running ${selectedMode.toUpperCase()} inference...`);

  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Request failed.");
    }

    setImageSrc(annotatedImage, data.annotated_image);
    modelSize.textContent = data.model_size || "-";
    runtimeDevice.textContent = data.device || "-";
    baselineMap.textContent = data.baseline_map || "";

    if (selectedMode === "anpr") {
      const anprDetections = data.results || data.detections || [];
      const detectionMs = Number(data.timings?.detection_ms ?? 0);
      const ocrMs = Number(data.timings?.ocr_ms ?? 0);
      const totalMs = Number(data.timings?.total_ms ?? 0);

      inferenceTime.textContent = `${totalMs.toFixed(2)} ms`;
      anprDetectionTime.textContent = `${detectionMs.toFixed(2)} ms`;
      anprOcrTime.textContent = `${ocrMs.toFixed(2)} ms`;
      anprTotalTime.textContent = `${totalMs.toFixed(2)} ms`;
      ocrBackend.textContent = data.ocr_backend || "-";
      renderAnprDetections(anprDetections);
    } else {
      inferenceTime.textContent = data.inference_time || "-";
      anprDetectionTime.textContent = "-";
      anprOcrTime.textContent = "-";
      anprTotalTime.textContent = "-";
      ocrBackend.textContent = "-";
      renderClassicDetections(data.detections);
    }

    latestInference.textContent = inferenceTime.textContent;
    latestRuntime.textContent = runtimeDevice.textContent;
    const count = selectedMode === "anpr" ? (data.results || data.detections || []).length : data.detections?.length || 0;
    latestDetections.textContent = String(count);

    const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean) : [];
    if (warnings.length > 0) {
      setStatus(`Done. ${count} ${mode.doneSuffix} ${warnings.join(" ")}`);
    } else {
      setStatus(`Done. ${count} ${mode.doneSuffix}`);
    }

    loadLeaderboardMetrics();
  } catch (error) {
    setStatus(`Error: ${error.message}`);
  } finally {
    detectBtn.disabled = false;
  }
});
