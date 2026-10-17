const API_BASE =
  window.location.protocol === "file:"
    ? "http://localhost:8000"
    : `${window.location.origin}`;

// DOM Elements
const imageInput = document.getElementById("imageInput");
const dropzone = document.getElementById("dropzone");
const dropzoneContent = document.getElementById("dropzoneContent");
const fileBadge = document.getElementById("fileBadge");
const fileNameText = document.getElementById("fileNameText");
const fileSizeText = document.getElementById("fileSizeText");
const fileBadgeIcon = document.getElementById("fileBadgeIcon");
const clearFileBtn = document.getElementById("clearFileBtn");
const fileLabel = document.getElementById("fileLabel");

const detectBtn = document.getElementById("detectBtn");
const previewImage = document.getElementById("previewImage");
const annotatedImage = document.getElementById("annotatedImage");
const previewVideo = document.getElementById("previewVideo");
const annotatedVideo = document.getElementById("annotatedVideo");
const previewFrame = document.getElementById("previewFrame");
const annotatedFrame = document.getElementById("annotatedFrame");
const detectionsHeading = document.getElementById("detectionsHeading");
const detectionList = document.getElementById("detectionList");

const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");
const ocrConfidenceWrap = document.getElementById("ocrConfidenceControl");
const ocrConfidenceSlider = document.getElementById("ocrConfidenceSlider");
const ocrConfidenceValue = document.getElementById("ocrConfidenceValue");

const frameStrideControl = document.getElementById("frameStrideControl");
const frameStrideSlider = document.getElementById("frameStrideSlider");
const frameStrideValue = document.getElementById("frameStrideValue");

const videoSummaryCard = document.getElementById("videoSummaryCard");
const videoPlatesTable = document.getElementById("videoPlatesTable");
const videoPlatesBody = document.getElementById("videoPlatesBody");
const uniqueCountBadge = document.getElementById("uniqueCountBadge");

const inferenceTime = document.getElementById("inferenceTime");
const modelSize = document.getElementById("modelSize");
const runtimeDevice = document.getElementById("runtimeDevice");
const latestDetectionsBadge = document.getElementById("latestDetectionsBadge");
const baselineMap = document.getElementById("baselineMap");

const statusWrap = document.getElementById("statusWrap");
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
const anprTimingsCard = document.getElementById("anprTimingsCard");
const anprDetectionTime = document.getElementById("anprDetectionTime");
const anprOcrTime = document.getElementById("anprOcrTime");
const anprTotalTime = document.getElementById("anprTotalTime");
const ocrBackend = document.getElementById("ocrBackend");

// State helper functions
const setFrameState = (img, hasImage) => {
  const frame = img?.parentElement;
  if (!frame) return;
  frame.classList.toggle("has-image", Boolean(hasImage));
};

const setVideoFrameState = (video, hasVideo) => {
  const frame = video?.parentElement;
  if (!frame) return;
  frame.classList.toggle("has-video", Boolean(hasVideo));
};

const setImageSrc = (img, src) => {
  if (!img) return;
  if (src) {
    img.src = src;
    img.style.display = "block";
    setFrameState(img, true);
  } else {
    img.removeAttribute("src");
    img.style.display = "none";
    setFrameState(img, false);
  }
};

const setVideoSrc = (video, src) => {
  if (!video) return;
  if (src) {
    video.src = src;
    video.style.display = "block";
    setVideoFrameState(video, true);
  } else {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.style.display = "none";
    setVideoFrameState(video, false);
  }
};

const formatFileSize = (bytes) => {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
};

const isVideoFile = (file) => {
  if (!file) return false;
  return (
    (file.type && file.type.startsWith("video/")) ||
    /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name || "")
  );
};

const showFileBadge = (file) => {
  if (!file || !fileBadge) return;
  const isVideo = isVideoFile(file);
  fileNameText.textContent = file.name;
  fileSizeText.textContent = formatFileSize(file.size);
  if (fileBadgeIcon) fileBadgeIcon.textContent = isVideo ? "🎥" : "🖼️";
  fileBadge.style.display = "flex";
  if (dropzoneContent) dropzoneContent.style.display = "none";
};

const hideFileBadge = () => {
  if (fileBadge) fileBadge.style.display = "none";
  if (dropzoneContent) dropzoneContent.style.display = "flex";
};

let selectedMode = "vehicle";

const modeUi = {
  vehicle: {
    endpoint: "/detect/vehicle",
    buttonText: "Detect Vehicles",
    icon: "🚗",
    heading: "Detected Vehicles",
    readyText: "Vehicle image ready. Click Detect Vehicles.",
    emptyText: "No vehicles detected in this image.",
    doneSuffix: "vehicle(s) detected.",
  },
  plate: {
    endpoint: "/detect/plate",
    buttonText: "Detect Plates",
    icon: "🔍",
    heading: "Detected Number Plates",
    readyText: "Plate image ready. Click Detect Plates.",
    emptyText: "No plates detected in this image.",
    doneSuffix: "plate(s) detected.",
  },
  anpr: {
    endpoint: "/anpr",
    buttonText: "Run Full ANPR",
    icon: "⚡",
    heading: "ANPR Plate + OCR Results",
    readyText: "Image ready. Click Run Full ANPR.",
    emptyText: "No vehicle/plate detections found for OCR in this image.",
    doneSuffix: "vehicle-linked plate result(s) processed.",
  },
  video: {
    endpoint: "/anpr/video",
    buttonText: "Process Video Stream",
    icon: "🎥",
    heading: "Video Stream Tracking Results",
    readyText: "Video stream ready. Click Process Video Stream.",
    emptyText: "No vehicles or plates detected in video.",
    doneSuffix: "unique vehicle(s) tracked and recognized.",
  },
};

const setStatus = (msg, state = "idle") => {
  if (statusText) statusText.textContent = msg;
  if (statusWrap) {
    statusWrap.className = `status-indicator status-${state}`;
  }
};

const asCheckText = (value, pass) => {
  const state = pass ? "PASS" : "FAIL";
  return `${value} (${state})`;
};

const loadLeaderboardMetrics = async () => {
  if (!leaderboardOverall) return;

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
      ? "Qualified (100% Passed)"
      : "Not qualified yet";
    leaderboardTimestamp.textContent = metrics.timestamp || "-";
  } catch (error) {
    leaderboardOverall.textContent = "Report not generated";
  }
};

// Navigation
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

// Slider Labels
const updateConfidenceLabel = () => {
  confidenceValue.textContent = Number(confidenceSlider.value).toFixed(2);
};

const updateOcrConfidenceLabel = () => {
  ocrConfidenceValue.textContent = Number(ocrConfidenceSlider.value).toFixed(2);
};

const updateFrameStrideLabel = () => {
  if (frameStrideValue && frameStrideSlider) {
    frameStrideValue.textContent = frameStrideSlider.value;
  }
};

updateConfidenceLabel();
updateOcrConfidenceLabel();
updateFrameStrideLabel();
confidenceSlider.addEventListener("input", updateConfidenceLabel);
ocrConfidenceSlider.addEventListener("input", updateOcrConfidenceLabel);
frameStrideSlider?.addEventListener("input", updateFrameStrideLabel);

// Mode Switching
const setDetectionMode = (mode) => {
  selectedMode = mode;
  modeButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));

  const btnText = detectBtn.querySelector(".btn-text");
  if (btnText) {
    btnText.textContent = modeUi[mode].buttonText;
  } else {
    detectBtn.textContent = modeUi[mode].buttonText;
  }

  const btnIcon = detectBtn.querySelector(".btn-icon");
  if (btnIcon) {
    btnIcon.textContent = modeUi[mode].icon || "⚡";
  }

  detectionsHeading.textContent = modeUi[mode].heading;
  if (ocrConfidenceWrap) {
    ocrConfidenceWrap.style.display = (mode === "anpr" || mode === "video") ? "flex" : "none";
  }
  if (frameStrideControl) {
    frameStrideControl.style.display = mode === "video" ? "flex" : "none";
  }
  if (anprTimingsCard) {
    anprTimingsCard.style.display = mode === "anpr" ? "flex" : "none";
  }
  if (videoSummaryCard) {
    videoSummaryCard.style.display = (mode === "video" && videoPlatesBody && videoPlatesBody.children.length > 0) ? "flex" : "none";
  }
  if (fileLabel) {
    fileLabel.innerHTML = mode === "video"
      ? `Drag & drop video stream, or <span class="highlight">Browse Files</span>`
      : `Drag & drop image or video here, or <span class="highlight">Browse Files</span>`;
  }
  setStatus(modeUi[mode].readyText, "idle");
  detectionList.innerHTML = "";
};

modeButtons.forEach((btn) => {
  btn.addEventListener("click", () => setDetectionMode(btn.dataset.mode));
});

setDetectionMode("vehicle");
loadLeaderboardMetrics();

// Drag-and-drop zone events
if (dropzone) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("drag-active");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("drag-active");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      imageInput.files = files;
      imageInput.dispatchEvent(new Event("change"));
    }
  });
}

clearFileBtn?.addEventListener("click", (e) => {
  e.stopPropagation();
  imageInput.value = "";
  hideFileBadge();
  setImageSrc(previewImage, "");
  setImageSrc(annotatedImage, "");
  setVideoSrc(previewVideo, "");
  setVideoSrc(annotatedVideo, "");
  detectionList.innerHTML = "";
  if (videoPlatesBody) videoPlatesBody.innerHTML = "";
  if (videoSummaryCard) videoSummaryCard.style.display = "none";
  if (latestDetectionsBadge) latestDetectionsBadge.textContent = "0 targets";
  inferenceTime.textContent = "-";
  modelSize.textContent = "-";
  runtimeDevice.textContent = "-";
  baselineMap.textContent = "";
  anprDetectionTime.textContent = "-";
  anprOcrTime.textContent = "-";
  anprTotalTime.textContent = "-";
  ocrBackend.textContent = "-";
  setStatus("Media cleared. Choose an image or video to begin.", "idle");
});

// File input change handler
imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) {
    hideFileBadge();
    return;
  }

  showFileBadge(file);
  const url = URL.createObjectURL(file);
  const isVideo = isVideoFile(file);

  if (isVideo) {
    if (selectedMode !== "video") {
      setDetectionMode("video");
    }
    setImageSrc(previewImage, "");
    setImageSrc(annotatedImage, "");
    setVideoSrc(annotatedVideo, "");
    setVideoSrc(previewVideo, url);
  } else {
    if (selectedMode === "video") {
      setDetectionMode("anpr");
    }
    setVideoSrc(previewVideo, "");
    setVideoSrc(annotatedVideo, "");
    setImageSrc(previewImage, url);
    setImageSrc(annotatedImage, "");
  }

  detectionList.innerHTML = "";
  if (videoPlatesBody) videoPlatesBody.innerHTML = "";
  if (videoSummaryCard) videoSummaryCard.style.display = "none";
  if (latestDetectionsBadge) latestDetectionsBadge.textContent = "0 targets";

  inferenceTime.textContent = "-";
  modelSize.textContent = "-";
  runtimeDevice.textContent = "-";
  baselineMap.textContent = "";
  anprDetectionTime.textContent = "-";
  anprOcrTime.textContent = "-";
  anprTotalTime.textContent = "-";
  ocrBackend.textContent = "-";

  setStatus(modeUi[selectedMode].readyText, "idle");
});

previewImage.addEventListener("error", () => setFrameState(previewImage, false));
annotatedImage.addEventListener("error", () => setFrameState(annotatedImage, false));
previewVideo?.addEventListener("error", () => setVideoFrameState(previewVideo, false));
annotatedVideo?.addEventListener("error", () => setVideoFrameState(annotatedVideo, false));

// Classical Detections (Vehicle or Plate only)
const renderClassicDetections = (detections) => {
  if (!detections || detections.length === 0) {
    detectionList.innerHTML = `<li>${modeUi[selectedMode].emptyText}</li>`;
    if (latestDetectionsBadge) latestDetectionsBadge.textContent = "0 targets";
    return;
  }

  if (latestDetectionsBadge) latestDetectionsBadge.textContent = `${detections.length} target(s)`;

  detections.forEach((det, index) => {
    const item = document.createElement("li");
    const confidencePct = Math.round(det.confidence * 100);
    const bbox = det.bbox.join(", ");
    item.innerHTML = `
      <div class="det-row">
        <span class="det-title">${index + 1}. ${det.label.toUpperCase()}</span>
        <span class="det-conf">${confidencePct}% CONF</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width: ${confidencePct}%"></div>
      </div>
      <div class="det-bbox"><strong>Bounding Box:</strong> [${bbox}]</div>
    `;
    detectionList.appendChild(item);
  });
};

// ANPR Plate + OCR Detections with Indian HSRP Plate Mockup
const renderAnprDetections = (detections) => {
  if (!detections || detections.length === 0) {
    detectionList.innerHTML = `<li>${modeUi[selectedMode].emptyText}</li>`;
    if (latestDetectionsBadge) latestDetectionsBadge.textContent = "0 targets";
    return;
  }

  if (latestDetectionsBadge) latestDetectionsBadge.textContent = `${detections.length} target(s)`;

  detections.forEach((det, index) => {
    const item = document.createElement("li");
    const plateText = det.plate_text || det.cleaned_text || det.plate || det.corrected_ocr || "(unreadable)";
    const rawText = det.raw_text || det.raw_ocr || "-";
    const method = det.ocr_method || det.method_used || det.variant_used || "CRNN BiLSTM";
    const detPct = Math.round(((det.det_confidence ?? det.confidence) || 0) * 100);
    const ocrPct = Math.round(((det.ocr_confidence ?? det.confidence) || 0) * 100);
    const plateBbox = (det.plate_bbox || det.bbox || []).join(", ");
    const vehicleBbox = (det.vehicle_bbox || []).join(", ");
    const perItemTime = det.inference_time || "-";
    const isIndian = det.indian_format_like !== undefined ? Boolean(det.indian_format_like) : Boolean(det.format_valid);

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
    const hasDebug = Boolean(originalBlock || cropBlock || thumbs);

    item.innerHTML = `
      <div class="det-row">
        <div class="det-title">
          <span>${index + 1}.</span>
          <div class="hsrp-badge">
            <div class="hsrp-ind-strip">
              <span>&#9881;</span>
              <span>IND</span>
            </div>
            <span class="hsrp-plate-number">${plateText}</span>
          </div>
        </div>
        <div class="det-conf">DET ${detPct}% &bull; OCR ${ocrPct}%</div>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width: ${ocrPct}%; background: ${isIndian ? 'linear-gradient(90deg, #10b981, #06b6d4)' : 'linear-gradient(90deg, #6366f1, #06b6d4)'};"></div>
      </div>
      <div class="det-bbox">
        <strong>Raw OCR:</strong> ${rawText} &bull; <strong>Backbone:</strong> ${method} &bull; <strong>Latency:</strong> ${perItemTime}
      </div>
      <div class="det-bbox">
        <strong>Plate Coordinates:</strong> [${plateBbox || "-"}] ${vehicleBbox ? `&bull; <strong>Vehicle Box:</strong> [${vehicleBbox}]` : ""}
      </div>
      <div class="det-bbox">
        <strong>MoRTH Format Verified:</strong> ${isIndian ? '<span class="badge-verified">VERIFIED HSRP</span>' : '<span class="badge-unverified">STANDARD</span>'}
      </div>
      ${hasDebug ? `
      <details class="debug-panel">
        <summary>View Multi-Pass Preprocessing Filters</summary>
        <div class="debug-grid">
          ${originalBlock}
          ${cropBlock}
          ${thumbs}
        </div>
      </details>` : ""}
    `;
    detectionList.appendChild(item);
  });
};

// Video Stream Detections with Table Log
const renderVideoDetections = (plates) => {
  detectionList.innerHTML = "";
  if (!videoPlatesBody || !videoSummaryCard) return;
  videoPlatesBody.innerHTML = "";

  if (!plates || plates.length === 0) {
    videoSummaryCard.style.display = "none";
    detectionList.innerHTML = `<li>No license plates detected in the video stream.</li>`;
    if (latestDetectionsBadge) latestDetectionsBadge.textContent = "0 plates";
    if (uniqueCountBadge) uniqueCountBadge.textContent = "0 Unique Plates";
    return;
  }

  videoSummaryCard.style.display = "flex";
  if (latestDetectionsBadge) latestDetectionsBadge.textContent = `${plates.length} vehicles`;
  if (uniqueCountBadge) uniqueCountBadge.textContent = `${plates.length} Unique Vehicles`;

  plates.forEach((plate, idx) => {
    const tr = document.createElement("tr");
    const confPct = Math.round((plate.confidence || 0) * 100);
    const isIndian = Boolean(plate.indian_format_like);
    const thumbHtml = plate.thumbnail
      ? `<img src="${plate.thumbnail}" alt="${plate.plate_text}" class="plate-thumb" />`
      : `<span style="color:var(--text-muted);">-</span>`;

    tr.innerHTML = `
      <td>${idx + 1}</td>
      <td>
        <div class="hsrp-badge">
          <div class="hsrp-ind-strip">
            <span>&#9881;</span>
            <span>IND</span>
          </div>
          <span class="hsrp-plate-number">${plate.plate_text}</span>
        </div>
      </td>
      <td>
        <div style="display:flex; align-items:center; gap:8px;">
          <strong style="color:var(--text-highlight); font-family:var(--font-mono);">${confPct}%</strong>
          <div class="conf-bar" style="width:60px; height:5px; margin:0;">
            <div class="conf-fill" style="width:${confPct}%;"></div>
          </div>
        </div>
      </td>
      <td style="font-family:var(--font-mono); font-size:0.82rem;">${plate.first_seen_sec}s - ${plate.last_seen_sec}s (${plate.duration_sec}s)</td>
      <td style="font-family:var(--font-mono);">${plate.frames_tracked} frames</td>
      <td>
        ${isIndian ? '<span class="badge-verified">VERIFIED</span>' : '<span class="badge-unverified">STANDARD</span>'}
      </td>
      <td>${thumbHtml}</td>
    `;
    videoPlatesBody.appendChild(tr);

    const li = document.createElement("li");
    li.innerHTML = `
      <div class="det-row">
        <div class="det-title">
          <span>#${plate.track_id}</span>
          <div class="hsrp-badge">
            <div class="hsrp-ind-strip">
              <span>&#9881;</span>
              <span>IND</span>
            </div>
            <span class="hsrp-plate-number">${plate.plate_text}</span>
          </div>
        </div>
        <div class="det-conf">${confPct}% &bull; tracked ${plate.frames_tracked}x</div>
      </div>
      <div class="conf-bar">
        <div class="conf-fill" style="width: ${confPct}%; background: ${isIndian ? 'linear-gradient(90deg, #10b981, #06b6d4)' : 'linear-gradient(90deg, #6366f1, #06b6d4)'};"></div>
      </div>
      <div class="det-bbox">
        <strong>Time Window:</strong> ${plate.first_seen_sec}s &rarr; ${plate.last_seen_sec}s (${plate.duration_sec}s duration)
      </div>
      <div class="det-bbox">
        <strong>MoRTH Verified:</strong> ${isIndian ? '<span class="badge-verified">YES - HSRP SYNTAX</span>' : '<span class="badge-unverified">STANDARD FORMAT</span>'}
      </div>
    `;
    detectionList.appendChild(li);
  });
};

// Safe JSON API response parser
const parseApiResponse = async (response, fallbackMsg) => {
  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (e) {
    if (!response.ok) {
      throw new Error(`Server error (${response.status}): ${text.slice(0, 150) || response.statusText || fallbackMsg}`);
    }
    throw new Error(`Invalid response format from server.`);
  }
  if (!response.ok) {
    throw new Error(data?.detail || fallbackMsg);
  }
  return data;
};

// Main Detect / Inference Button
detectBtn.addEventListener("click", async () => {
  const file = imageInput.files?.[0];
  if (!file) {
    setStatus(selectedMode === "video" ? "Please upload a video file first." : "Please upload an image first.", "error");
    return;
  }

  const isVideo = isVideoFile(file);

  // Auto-switch mode if there is a mismatch
  if (isVideo && selectedMode !== "video") {
    setDetectionMode("video");
  } else if (!isVideo && selectedMode === "video") {
    setDetectionMode("anpr");
  }

  const mode = modeUi[selectedMode];
  const endpoint = `${API_BASE}${mode.endpoint}`;

  const formData = new FormData();
  formData.append("file", file);

  if (selectedMode === "video") {
    formData.append("det_confidence", confidenceSlider.value);
    formData.append("ocr_confidence", ocrConfidenceSlider.value);
    formData.append("frame_stride", frameStrideSlider ? frameStrideSlider.value : "2");

    detectBtn.disabled = true;
    detectionList.innerHTML = "";
    if (videoPlatesBody) videoPlatesBody.innerHTML = "";
    setStatus("Analyzing video stream on CUDA GPU... Tracking multi-frame plates.", "loading");

    try {
      const response = await fetch(endpoint, { method: "POST", body: formData });
      const data = await parseApiResponse(response, "Video processing failed.");

      setVideoSrc(annotatedVideo, `${API_BASE}${data.video_url}`);
      modelSize.textContent = "YOLO11n + CRNN";
      runtimeDevice.textContent = "cuda:0";
      inferenceTime.textContent = `${data.processing_time_sec}s (${data.processing_fps} fps)`;
      baselineMap.textContent = `Analyzed ${data.processed_frames} of ${data.total_frames} frames (${data.video_duration_sec}s stream @ ${data.video_fps} fps)`;

      renderVideoDetections(data.unique_plates || []);
      const count = data.unique_vehicles_count ?? (data.unique_plates?.length || 0);
      if (latestDetections) latestDetections.textContent = String(count);
      if (latestInference) latestInference.textContent = inferenceTime.textContent;
      if (latestRuntime) latestRuntime.textContent = runtimeDevice.textContent;
      setStatus(`Completed. Successfully tracked ${count} unique vehicle(s)!`, "success");
      loadLeaderboardMetrics();
    } catch (error) {
      setStatus(`Error: ${error.message}`, "error");
    } finally {
      detectBtn.disabled = false;
    }
    return;
  }

  if (selectedMode === "anpr") {
    formData.append("det_confidence", confidenceSlider.value);
    formData.append("ocr_confidence", ocrConfidenceSlider.value);
  } else {
    formData.append("confidence", confidenceSlider.value);
  }

  detectBtn.disabled = true;
  detectionList.innerHTML = "";
  setStatus(`Running ${selectedMode.toUpperCase()} neural inference on CUDA...`, "loading");

  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const data = await parseApiResponse(response, "Detection request failed.");

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
      ocrBackend.textContent = data.ocr_backend || "CRNN BiLSTM + CTC";
      renderAnprDetections(anprDetections);
    } else {
      inferenceTime.textContent = data.inference_time || "-";
      anprDetectionTime.textContent = "-";
      anprOcrTime.textContent = "-";
      anprTotalTime.textContent = "-";
      ocrBackend.textContent = "-";
      renderClassicDetections(data.detections);
    }

    if (latestInference) latestInference.textContent = inferenceTime.textContent;
    if (latestRuntime) latestRuntime.textContent = runtimeDevice.textContent;
    const count = selectedMode === "anpr" ? (data.results || data.detections || []).length : data.detections?.length || 0;
    if (latestDetections) latestDetections.textContent = String(count);

    const warnings = Array.isArray(data.warnings) ? data.warnings.filter(Boolean) : [];
    if (warnings.length > 0) {
      setStatus(`Done. ${count} ${mode.doneSuffix} ${warnings.join(" ")}`, "success");
    } else {
      setStatus(`Done. ${count} ${mode.doneSuffix}`, "success");
    }

    loadLeaderboardMetrics();
  } catch (error) {
    setStatus(`Error: ${error.message}`, "error");
  } finally {
    detectBtn.disabled = false;
  }
});
