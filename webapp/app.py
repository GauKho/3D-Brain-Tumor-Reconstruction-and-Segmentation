import shutil
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.inference import (
    ModelService,
    build_meshes,
    build_preview,
    build_result_slice,
    load_case,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "webapp" / "static"
RUNTIME_DIR = ROOT / "runtime" / "cases"
MODEL_SPECS = OrderedDict(
    [
        (
            "latup-net-wavelet",
            {
                "name": "Latup-net-wavelet",
                "description": "LATUPNet 9 kênh với wavelet và trọng số loss cố định.",
                "checkpoint": ROOT
                / "outputs"
                / "best_latupnet_wavelet_region3_sigmoid_priority1.pth",
            },
        ),
        (
            "rel-ppo",
            {
                "name": "Rel_ppo",
                "description": "LATUPNet 9 kênh được huấn luyện cùng bộ điều khiển PPO.",
                "checkpoint": ROOT
                / "REL"
                / "working"
                / "working"
                / "best_model_rl_rel_ppo.pth",
            },
        ),
    ]
)
DEFAULT_MODEL_ID = "latup-net-wavelet"
CHECKPOINT = MODEL_SPECS[DEFAULT_MODEL_ID]["checkpoint"]
SAMPLE_DIR = ROOT / "BraTS20_Validation_031_t1ce.nii"
MODALITIES = ("t1ce", "t2", "flair")
MAX_UPLOAD_BYTES = 160 * 1024 * 1024

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="TumorMesh Multi-Model Studio")
model_services = {
    model_id: ModelService(
        spec["checkpoint"],
        model_id=model_id,
        model_name=spec["name"],
    )
    for model_id, spec in MODEL_SPECS.items()
}
cases = OrderedDict()
cases_lock = threading.Lock()
inference_lock = threading.Lock()


def register_case(case):
    with cases_lock:
        cases[case.case_id] = case
        cases.move_to_end(case.case_id)
        while len(cases) > 2:
            cases.popitem(last=False)


def get_case(case_id):
    with cases_lock:
        case = cases.get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Ca MRI không còn trong phiên làm việc.")
    return case


def valid_nifti_name(filename):
    lower = (filename or "").lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


async def save_upload(upload, destination):
    if not valid_nifti_name(upload.filename):
        raise HTTPException(status_code=400, detail=f"{upload.filename}: chỉ chấp nhận .nii hoặc .nii.gz")
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                output.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Một file NIfTI vượt quá giới hạn 160 MB.")
            output.write(chunk)
    await upload.close()


@app.get("/api/status")
def status():
    sample_files = [next(SAMPLE_DIR.glob(f"*_{modality}.nii*"), None) for modality in MODALITIES] if SAMPLE_DIR.is_dir() else []
    models = [
        {
            "id": model_id,
            "name": spec["name"],
            "description": spec["description"],
            "ready": spec["checkpoint"].exists(),
            "loaded": model_services[model_id].loaded,
        }
        for model_id, spec in MODEL_SPECS.items()
    ]
    return {
        "checkpoint_ready": any(model["ready"] for model in models),
        "all_models_ready": all(model["ready"] for model in models),
        "model_loaded": any(model["loaded"] for model in models),
        "default_model_id": DEFAULT_MODEL_ID,
        "models": models,
        "sample_available": len(sample_files) == 3 and all(sample_files),
    }


@app.post("/api/cases")
async def create_case(
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    flair: UploadFile = File(...),
):
    case_id = uuid.uuid4().hex[:12]
    case_dir = RUNTIME_DIR / case_id
    case_dir.mkdir(parents=True)
    uploads = {"t1ce": t1ce, "t2": t2, "flair": flair}
    paths = {}
    try:
        for modality, upload in uploads.items():
            suffix = ".nii.gz" if upload.filename.lower().endswith(".nii.gz") else ".nii"
            destination = case_dir / f"{modality}{suffix}"
            await save_upload(upload, destination)
            paths[modality] = destination
        case_name = Path(t1ce.filename).name.split("_t1ce")[0] or f"case-{case_id}"
        case = load_case(case_id, case_name, paths)
        register_case(case)
        return build_preview(case)
    except HTTPException:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(case_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Không thể đọc bộ NIfTI: {error}") from error


@app.post("/api/sample")
def load_sample():
    if not SAMPLE_DIR.is_dir():
        raise HTTPException(status_code=404, detail="Không tìm thấy ca mẫu.")
    paths = {}
    for modality in MODALITIES:
        path = next(SAMPLE_DIR.glob(f"*_{modality}.nii*"), None)
        if path is None:
            raise HTTPException(status_code=404, detail=f"Ca mẫu thiếu modality {modality.upper()}.")
        paths[modality] = path
    case_id = uuid.uuid4().hex[:12]
    case_name = paths["t1ce"].name.split("_t1ce")[0]
    try:
        case = load_case(case_id, case_name, paths)
        register_case(case)
        return build_preview(case)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Không thể đọc ca mẫu: {error}") from error


@app.get("/api/cases/{case_id}/slice")
def case_slice(case_id: str, index: int):
    return build_preview(get_case(case_id), index=index)


@app.post("/api/cases/{case_id}/diagnose")
def diagnose(case_id: str, model_id: str = DEFAULT_MODEL_ID):
    case = get_case(case_id)
    if model_id not in MODEL_SPECS:
        available = ", ".join(MODEL_SPECS)
        raise HTTPException(
            status_code=400,
            detail=f"Model không hợp lệ. Các lựa chọn: {available}.",
        )
    spec = MODEL_SPECS[model_id]
    if not spec["checkpoint"].exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model {spec['name']} chưa có checkpoint trên máy.",
        )
    try:
        with inference_lock:
            return model_services[model_id].diagnose(case)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise HTTPException(status_code=507, detail="GPU không đủ bộ nhớ để chạy volume 128³.") from error
        raise HTTPException(status_code=500, detail=f"Inference thất bại: {error}") from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Inference thất bại: {error}") from error


@app.get("/api/cases/{case_id}/result-slice")
def result_slice(case_id: str, axis: str, index: int):
    case = get_case(case_id)
    try:
        return build_result_slice(case, axis.lower(), index)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/cases/{case_id}/mesh")
def result_mesh(case_id: str):
    case = get_case(case_id)
    try:
        return build_meshes(case)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/cases/{case_id}/prediction")
def download_prediction(case_id: str):
    case = get_case(case_id)
    if case.prediction_path is None or not case.prediction_path.exists():
        raise HTTPException(status_code=404, detail="Ca này chưa có kết quả dự đoán.")
    return FileResponse(
        case.prediction_path,
        media_type="application/gzip",
        filename=f"pred_{case.case_name}_{case.model_id or 'segmentation'}.nii.gz",
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
