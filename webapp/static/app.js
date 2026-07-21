const state = {
  files: { t1ce: null, t2: null, flair: null },
  caseId: null,
  preview: null,
  result: null,
  viewer: null,
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

async function loadStatus() {
  try {
    const status = await request("/api/status");
    els.statusDot.className = `status-dot ${status.checkpoint_ready ? "ready" : "error"}`;
    els.statusText.textContent = status.checkpoint_ready ? "Studio sẵn sàng" : "Thiếu mô hình";
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
  if (state.viewer?.dispose) state.viewer.dispose();
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
  els.diagnoseButton.disabled = false;
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

const progressSteps = [
  ["Đang chuẩn hóa volume", "Resample 1 mm, center crop/pad, clip P1–P99 và masked Min-Max"],
  ["Đang tạo đặc trưng wavelet", "Symlet-8 level 2 tạo sáu energy map từ ba nguồn MRI"],
  ["LATUPNet đang phân tích", "Tensor 9 kênh được suy luận thành ba vùng WT, TC và ET"],
  ["Đang khôi phục mask", "Đảo crop/pad, resample nearest-neighbor về NIfTI gốc"],
  ["Đang kiểm tra kết quả", "Ép ET ⊆ TC ⊆ WT, khóa brain mask và tạo output 2D/3D"],
];

function startProgress() {
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
  els.diagnoseButton.disabled = true;
  els.progressSection.classList.remove("hidden");
  els.resultsSection.classList.add("hidden");
  els.progressSection.scrollIntoView({ behavior: "smooth", block: "center" });
  const progressTimer = startProgress();
  try {
    const result = await request(`/api/cases/${state.caseId}/diagnose`, { method: "POST" });
    state.result = result;
    renderResult(result);
    showToast("LATUPNet đã hoàn tất phân đoạn và kiểm tra brain mask.");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    clearInterval(progressTimer);
    els.progressSection.classList.add("hidden");
    els.diagnoseButton.disabled = false;
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
  if (state.viewer?.dispose) state.viewer.dispose();
  const stage = $("viewerStage");
  const canvas = $("segmentationCanvas");
  stage.querySelector(".viewer-error")?.remove();
  try {
    const THREE = await import("/vendor/three.module.js");
    const { OrbitControls } = await import("/vendor/OrbitControls.js");
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xe9efec, 1);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    camera.position.set(2.8, 2.1, 3.1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
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
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.renderOrder = isBrain ? 0 : 1;
      scene.add(mesh);
      meshByLabel.set(String(data.label), mesh);
    });

    if (!meshByLabel.has("brain")) throw new Error("Không tạo được bề mặt não từ brain mask.");

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

    let frameId;
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    document.querySelectorAll("#classToggles input").forEach(input => input.addEventListener("change", () => {
      const mesh = meshByLabel.get(input.dataset.label);
      if (mesh) mesh.visible = input.checked;
    }));
    $("opacityRange").oninput = (event) => {
      const opacity = Number(event.target.value) / 100;
      $("opacityOutput").textContent = `${event.target.value}%`;
      meshByLabel.forEach((mesh, label) => {
        if (label !== "brain") mesh.material.opacity = opacity;
      });
    };
    $("rotateToggle").onchange = (event) => { controls.autoRotate = event.target.checked; };
    $("wireframeToggle").onchange = (event) => meshByLabel.forEach(mesh => { mesh.material.wireframe = event.target.checked; });
    $("resetViewButton").onclick = () => {
      camera.position.set(2.8, 2.1, 3.1);
      controls.target.set(0, 0, 0);
      controls.update();
    };

    $("viewerLoading").classList.add("hidden");
    state.viewer = {
      dispose() {
        cancelAnimationFrame(frameId);
        observer.disconnect();
        controls.dispose();
        meshByLabel.forEach(mesh => { mesh.geometry.dispose(); mesh.material.dispose(); });
        renderer.dispose();
      },
    };
  } catch (error) {
    showViewerError(error);
  }
}

els.sampleButton.addEventListener("click", loadSample);
els.diagnoseButton.addEventListener("click", diagnose);
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
