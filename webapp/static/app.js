const state = {
  files: { t1ce: null, t2: null, flair: null },
  caseId: null,
  preview: null,
  result: null,
  viewer: null,
  models: [],
  selectedModelId: null,
  isDiagnosing: false,
  sliceTimer: null,
  resultSliceTimers: {},
  resultSliceRequests: {},
};

const $ = (id) => document.getElementById(id);
const els = {
  statusDot: $("statusDot"), statusText: $("statusText"),
  sampleButton: $("sampleButton"), uploadMessage: $("uploadMessage"), diagnoseButton: $("diagnoseButton"),
  emptyState: $("emptyState"), previewContent: $("previewContent"), caseTitle: $("caseTitle"),
  volumeMeta: $("volumeMeta"), sliceRange: $("sliceRange"), sliceOutput: $("sliceOutput"),
  progressSection: $("progressSection"), resultsSection: $("resultsSection"),
  progressTitle: $("progressTitle"), progressText: $("progressText"), toast: $("toast"),
  modelPickerButton: $("modelPickerButton"), modelMenu: $("modelMenu"),
  selectedModelName: $("selectedModelName"), selectedModelDescription: $("selectedModelDescription"),
};

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 } });
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Lỗi ${response.status}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

function showToast(message, error = false) {
  els.toast.textContent = message;
  els.toast.className = `toast show${error ? " error" : ""}`;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.className = "toast", 3600);
}

function setUploadMessage(message, mode = "") {
  els.uploadMessage.className = `upload-message ${mode}`.trim();
  els.uploadMessage.querySelector("span").textContent = message;
}

function getSelectedModel() {
  return state.models.find(model => model.id === state.selectedModelId) || null;
}

function refreshDiagnoseAvailability() {
  const model = getSelectedModel();
  els.diagnoseButton.disabled = state.isDiagnosing || !state.caseId || !model?.ready;
  els.modelPickerButton.disabled = state.isDiagnosing || !state.models.some(item => item.ready);
}

function setModelMenuOpen(open) {
  const shouldOpen = Boolean(open) && !els.modelPickerButton.disabled;
  els.modelMenu.classList.toggle("hidden", !shouldOpen);
  els.modelPickerButton.setAttribute("aria-expanded", String(shouldOpen));
}

function renderModelOptions() {
  els.modelMenu.innerHTML = state.models.map(model => {
    const selected = model.id === state.selectedModelId;
    const detail = model.ready ? model.description : "Không tìm thấy model trên máy";
    return `
      <button
        class="model-option${selected ? " selected" : ""}"
        type="button"
        role="option"
        aria-selected="${selected}"
        data-model-id="${model.id}"
        ${model.ready ? "" : "disabled"}
      >
        <span class="model-option-icon" aria-hidden="true"><i data-lucide="network"></i></span>
        <span class="model-option-copy"><strong>${model.name}</strong><small>${detail}</small></span>
        <i class="model-option-check" data-lucide="check" aria-hidden="true"></i>
      </button>
    `;
  }).join("");

  els.modelMenu.querySelectorAll(".model-option:not(:disabled)").forEach(option => {
    option.addEventListener("click", () => selectModel(option.dataset.modelId, true));
  });
  refreshIcons();
}

function selectModel(modelId, announce = false) {
  const model = state.models.find(item => item.id === modelId && item.ready);
  if (!model) return;
  state.selectedModelId = model.id;
  els.selectedModelName.textContent = model.name;
  els.selectedModelDescription.textContent = model.description;
  renderModelOptions();
  setModelMenuOpen(false);
  refreshDiagnoseAvailability();
  if (announce) showToast(`Đã chọn ${model.name}.`);
}

async function loadStatus() {
  try {
    const status = await request("/api/status");
    state.models = status.models || [];
    const readyModels = state.models.filter(model => model.ready);
    const preferredId = readyModels.some(model => model.id === state.selectedModelId)
      ? state.selectedModelId
      : (readyModels.find(model => model.id === status.default_model_id)?.id || readyModels[0]?.id);
    if (preferredId) {
      selectModel(preferredId);
    } else {
      state.selectedModelId = null;
      els.selectedModelName.textContent = "Không có model khả dụng";
      els.selectedModelDescription.textContent = "Kiểm tra lại các tệp model của dự án";
      renderModelOptions();
      refreshDiagnoseAvailability();
    }
    els.statusDot.className = `status-dot ${readyModels.length ? "ready" : "error"}`;
    els.statusText.textContent = readyModels.length
      ? `${readyModels.length}/${state.models.length} mô hình sẵn sàng`
      : "Thiếu mô hình";
    els.sampleButton.disabled = !status.sample_available;
  } catch (error) {
    els.statusDot.className = "status-dot error";
    els.statusText.textContent = "Backend chưa sẵn sàng";
    showToast(error.message, true);
  }
}

function bindFileInput(modality) {
  const input = $(`${modality}Input`);
  input.addEventListener("change", () => {
    const file = input.files[0] || null;
    state.files[modality] = file;
    $(`${modality}Name`).textContent = file ? file.name : "Chọn file NIfTI";
    input.closest(".file-row").classList.toggle("loaded", Boolean(file));
    if (Object.values(state.files).every(Boolean)) uploadFiles();
  });
}

async function uploadFiles() {
  els.diagnoseButton.disabled = true;
  setUploadMessage("Đang đọc và kiểm tra ba volume MRI...");
  const form = new FormData();
  Object.entries(state.files).forEach(([key, file]) => form.append(key, file));
  try {
    const preview = await request("/api/cases", { method: "POST", body: form });
    applyPreview(preview);
    setUploadMessage("Đã nạp đủ ba modality. Có thể chạy chẩn đoán.", "success");
  } catch (error) {
    setUploadMessage(error.message, "error");
    showToast(error.message, true);
  }
}

async function loadSample() {
  els.sampleButton.disabled = true;
  setUploadMessage("Đang nạp ca BraTS mẫu...");
  try {
    const preview = await request("/api/sample", { method: "POST" });
    applyPreview(preview);
    setUploadMessage("Ca mẫu đã sẵn sàng để chẩn đoán.", "success");
  } catch (error) {
    setUploadMessage(error.message, "error");
    showToast(error.message, true);
  } finally {
    els.sampleButton.disabled = false;
  }
}

function applyPreview(preview) {
  state.caseId = preview.case_id;
  state.preview = preview;
  state.result = null;
  if (state.viewer?.suspend) state.viewer.suspend();
  els.caseTitle.textContent = preview.case_name;
  els.volumeMeta.textContent = `${preview.shape.join(" × ")} voxel · spacing ${preview.spacing.join(" × ")} mm`;
  els.emptyState.classList.add("hidden");
  els.previewContent.classList.remove("hidden");
  els.resultsSection.classList.add("hidden");
  $("t1cePreview").src = preview.slices.t1ce;
  $("t2Preview").src = preview.slices.t2;
  $("flairPreview").src = preview.slices.flair;
  els.sliceRange.max = preview.slice_max;
  els.sliceRange.value = preview.slice_index;
  els.sliceOutput.textContent = `${preview.slice_index} / ${preview.slice_max}`;
  refreshDiagnoseAvailability();
}

async function updateSlice(index) {
  if (!state.caseId) return;
  try {
    const preview = await request(`/api/cases/${state.caseId}/slice?index=${index}`);
    $("t1cePreview").src = preview.slices.t1ce;
    $("t2Preview").src = preview.slices.t2;
    $("flairPreview").src = preview.slices.flair;
  } catch (error) {
    showToast(error.message, true);
  }
}

function startProgress(modelName) {
  const progressSteps = [
    ["Đang chuẩn hóa volume", "Resample 1 mm, center crop/pad, clip P1–P99 và masked Min-Max"],
    ["Đang tạo đặc trưng wavelet", "Symlet-8 level 2 tạo sáu energy map từ ba nguồn MRI"],
    [`${modelName} đang phân tích`, "Tensor 9 kênh được suy luận thành ba vùng WT, TC và ET"],
    ["Đang khôi phục mask", "Đảo crop/pad, resample nearest-neighbor về NIfTI gốc"],
    ["Đang kiểm tra kết quả", "Ép ET ⊆ TC ⊆ WT, khóa brain mask và tạo output 2D/3D"],
  ];
  let index = 0;
  const update = () => {
    els.progressTitle.textContent = progressSteps[index][0];
    els.progressText.textContent = progressSteps[index][1];
    index = Math.min(index + 1, progressSteps.length - 1);
  };
  update();
  return setInterval(update, 1800);
}

async function diagnose() {
  if (!state.caseId) return;
  const selectedModel = getSelectedModel();
  if (!selectedModel?.ready) {
    showToast("Hãy chọn một model khả dụng trước khi chẩn đoán.", true);
    return;
  }
  state.isDiagnosing = true;
  refreshDiagnoseAvailability();
  setModelMenuOpen(false);
  if (state.viewer?.suspend) state.viewer.suspend();
  els.progressSection.classList.remove("hidden");
  els.resultsSection.classList.add("hidden");
  els.progressSection.scrollIntoView({ behavior: "smooth", block: "center" });
  const progressTimer = startProgress(selectedModel.name);
  try {
    const modelId = encodeURIComponent(selectedModel.id);
    const result = await request(`/api/cases/${state.caseId}/diagnose?model_id=${modelId}`, { method: "POST" });
    state.result = result;
    renderResult(result);
    showToast(`${result.model_name} đã hoàn tất phân đoạn và kiểm tra brain mask.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    clearInterval(progressTimer);
    els.progressSection.classList.add("hidden");
    state.isDiagnosing = false;
    refreshDiagnoseAvailability();
  }
}

function applyResultSlice(data) {
  $(`${data.axis}Result`).src = data.image;
  const range = $(`${data.axis}Range`);
  range.max = data.slice_max;
  range.value = data.index;
  $(`${data.axis}Output`).textContent = `${data.index} / ${data.slice_max}`;
}

async function updateResultSlice(axis, index) {
  const requestId = (state.resultSliceRequests[axis] || 0) + 1;
  state.resultSliceRequests[axis] = requestId;
  try {
    const data = await request(`/api/cases/${state.caseId}/result-slice?axis=${axis}&index=${index}`);
    if (state.resultSliceRequests[axis] === requestId) applyResultSlice(data);
  } catch (error) {
    showToast(error.message, true);
  }
}

function renderResult(result) {
  $("resultModelName").textContent = result.model_name;
  $("volumeMetric").textContent = `${result.tumor_volume_ml.toFixed(3)} ml`;
  $("confidenceMetric").textContent = `${result.mean_confidence.toFixed(1)}%`;
  $("ratioMetric").textContent = `${(result.tumor_to_brain_ratio * 100).toFixed(2)}%`;
  $("timeMetric").textContent = `${result.inference_seconds.toFixed(2)} s`;
  $("downloadButton").href = result.download_url;

  Object.values(result.default_slices).forEach(applyResultSlice);
  $("classBreakdown").innerHTML = result.class_stats.map(item => `
    <div class="breakdown-row">
      <span class="swatch" style="background:${item.color}"></span>
      <span>${item.name} · ${item.voxels.toLocaleString("vi-VN")} voxel</span>
      <strong>${item.volume_ml.toFixed(3)} ml</strong>
    </div>
  `).join("");

  const qualityBanner = $("qualityBanner");
  qualityBanner.className = `quality-banner${result.quality_warning ? " warning" : ""}`;
  $("qualityTitle").textContent = result.quality_warning ? "Cần kiểm tra lại kết quả" : "Kiểm tra không gian hợp lệ";
  $("qualityMessage").textContent = `${result.quality_message} Voxel ngoài não: ${result.outside_brain_voxels.toLocaleString("vi-VN")}.`;

  const layers = [
    { label: "brain", name: "Não", color: "#9aa7a1" },
    ...result.class_stats,
  ];
  $("classToggles").innerHTML = layers.map(item => `
    <label class="class-toggle">
      <span class="swatch" style="background:${item.color}"></span>
      <span>${item.name}</span>
      <input type="checkbox" data-label="${item.label}" checked>
    </label>
  `).join("");

  els.resultsSection.classList.remove("hidden");
  $("viewerLoading").classList.remove("hidden");
  $("segmentationCanvas").style.display = "block";
  refreshIcons();
  els.resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  initViewerFromUrl(result.mesh_url);
}

async function initViewerFromUrl(meshUrl) {
  try {
    const payload = await request(meshUrl);
    await initViewer(payload.meshes);
  } catch (error) {
    showViewerError(error);
  }
}

function showViewerError(error) {
  const stage = $("viewerStage");
  $("viewerLoading").classList.add("hidden");
  $("segmentationCanvas").style.display = "none";
  stage.querySelector(".viewer-error")?.remove();
  const fallback = document.createElement("div");
  fallback.className = "viewer-error";
  fallback.innerHTML = `<div><strong>Không thể dựng mô hình 3D</strong><p>${error.message}</p><p>Mask NIfTI và ba overlay 2D vẫn khả dụng.</p></div>`;
  stage.appendChild(fallback);
}

async function initViewer(meshes) {
  const stage = $("viewerStage");
  stage.querySelector(".viewer-error")?.remove();

  try {
    if (state.viewer?.isContextLost()) {
      state.viewer.dispose();
      state.viewer = null;
      replaceViewerCanvas();
    }
    if (!state.viewer) state.viewer = await createViewerSession(stage, $("segmentationCanvas"));
    state.viewer.setMeshes(meshes);
    state.viewer.resume();
    $("viewerLoading").classList.add("hidden");
  } catch (error) {
    if (state.viewer?.dispose) state.viewer.dispose();
    state.viewer = null;
    if (isRecoverableWebGLError(error)) {
      try {
        const canvas = replaceViewerCanvas();
        await new Promise(resolve => requestAnimationFrame(resolve));
        state.viewer = await createViewerSession(stage, canvas);
        state.viewer.setMeshes(meshes);
        state.viewer.resume();
        $("viewerLoading").classList.add("hidden");
        return;
      } catch (retryError) {
        showViewerError(normalizeViewerError(retryError));
        return;
      }
    }
    showViewerError(normalizeViewerError(error));
  }
}

function replaceViewerCanvas() {
  const current = $("segmentationCanvas");
  const replacement = current.cloneNode(false);
  replacement.style.display = "block";
  current.replaceWith(replacement);
  return replacement;
}

function isRecoverableWebGLError(error) {
  return /precision|webgl|context/i.test(error?.message || "");
}

function normalizeViewerError(error) {
  if (isRecoverableWebGLError(error)) {
    return new Error("WebGL không thể khởi tạo lại sau inference. Hãy tải lại trang và kiểm tra Hardware Acceleration của trình duyệt.");
  }
  return error;
}

async function createViewerSession(stage, canvas) {
  const THREE = await import("/vendor/three.module.js");
  const { OrbitControls } = await import("/vendor/OrbitControls.js");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: false,
  });
  if (renderer.getContext().isContextLost()) {
    renderer.dispose();
    throw new Error("WebGL context was lost before the 3D scene initialized.");
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xe9efec, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
  camera.position.set(2.8, 2.1, 3.1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = $("rotateToggle").checked;
  controls.autoRotateSpeed = 1.15;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x466158, 2.0));
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
  keyLight.position.set(3, 4, 5);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x7fc8ba, 1.8);
  rimLight.position.set(-4, 1, -2);
  scene.add(rimLight);

  const meshByLabel = new Map();
  let frameId = null;
  let running = false;

  const clearMeshes = () => {
    meshByLabel.forEach(mesh => {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    });
    meshByLabel.clear();
    renderer.renderLists.dispose();
  };

  const setMeshes = meshes => {
    clearMeshes();
    meshes.forEach(data => {
      if (!data.vertices.length || !data.faces.length) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.vertices, 3));
      geometry.setIndex(data.faces);
      geometry.computeVertexNormals();
      geometry.computeBoundingSphere();
      const isBrain = String(data.label) === "brain";
      const material = new THREE.MeshStandardMaterial({
        color: data.color,
        roughness: isBrain ? 0.72 : 0.46,
        metalness: 0.02,
        transparent: true,
        opacity: data.opacity,
        depthWrite: !isBrain,
        wireframe: $("wireframeToggle").checked,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = isBrain ? 0 : 1;
      scene.add(mesh);
      meshByLabel.set(String(data.label), mesh);
    });
    if (!meshByLabel.has("brain")) throw new Error("Không tạo được bề mặt não từ brain mask.");

    document.querySelectorAll("#classToggles input").forEach(input => {
      input.onchange = () => {
        const mesh = meshByLabel.get(input.dataset.label);
        if (mesh) mesh.visible = input.checked;
      };
    });
  };

  const resize = () => {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(stage);
  resize();

  const animate = () => {
    if (!running) return;
    controls.update();
    renderer.render(scene, camera);
    frameId = requestAnimationFrame(animate);
  };
  const resume = () => {
    if (running) return;
    running = true;
    animate();
  };
  const suspend = () => {
    running = false;
    if (frameId !== null) cancelAnimationFrame(frameId);
    frameId = null;
    clearMeshes();
    renderer.clear();
  };

  $("opacityRange").oninput = event => {
    const opacity = Number(event.target.value) / 100;
    $("opacityOutput").textContent = `${event.target.value}%`;
    meshByLabel.forEach((mesh, label) => {
      if (label !== "brain") mesh.material.opacity = opacity;
    });
  };
  $("rotateToggle").onchange = event => { controls.autoRotate = event.target.checked; };
  $("wireframeToggle").onchange = event => meshByLabel.forEach(mesh => { mesh.material.wireframe = event.target.checked; });
  $("resetViewButton").onclick = () => {
    camera.position.set(2.8, 2.1, 3.1);
    controls.target.set(0, 0, 0);
    controls.update();
  };

  return {
    setMeshes,
    resume,
    suspend,
    isContextLost() {
      return renderer.getContext().isContextLost();
    },
    dispose() {
      suspend();
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
    },
  };
}

els.sampleButton.addEventListener("click", loadSample);
els.diagnoseButton.addEventListener("click", diagnose);
els.modelPickerButton.addEventListener("click", () => {
  setModelMenuOpen(els.modelPickerButton.getAttribute("aria-expanded") !== "true");
});
document.addEventListener("click", event => {
  if (!event.target.closest(".model-picker")) setModelMenuOpen(false);
});
document.addEventListener("keydown", event => {
  if (event.key === "Escape") setModelMenuOpen(false);
});
els.sliceRange.addEventListener("input", (event) => {
  els.sliceOutput.textContent = `${event.target.value} / ${event.target.max}`;
  clearTimeout(state.sliceTimer);
  state.sliceTimer = setTimeout(() => updateSlice(event.target.value), 120);
});
["axial", "coronal", "sagittal"].forEach(axis => {
  $(`${axis}Range`).addEventListener("input", event => {
    $(`${axis}Output`).textContent = `${event.target.value} / ${event.target.max}`;
    clearTimeout(state.resultSliceTimers[axis]);
    state.resultSliceTimers[axis] = setTimeout(() => updateResultSlice(axis, event.target.value), 120);
  });
});
["t1ce", "t2", "flair"].forEach(bindFileInput);
refreshIcons();
loadStatus();
