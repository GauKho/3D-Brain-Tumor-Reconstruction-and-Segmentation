import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import nibabel as nib
import numpy as np
import pywt
import torch
from PIL import Image
from scipy.ndimage import zoom as nd_zoom
from skimage.measure import marching_cubes
from skimage.transform import resize as sk_resize

from webapp.model import LATUPNet


TARGET_SHAPE = (128, 128, 128)
TARGET_SPACING = (1.0, 1.0, 1.0)
PERCENTILE_RANGE = (1.0, 99.0)
WAVELET_NAME = "sym8"
WAVELET_LEVEL = 2
WAVELET_MODE = "symmetric"
MODEL_INPUT_CHANNELS = 9
MODEL_OUTPUT_REGIONS = 3
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}
LABELS = {
    1: {"name": "NCR/NET", "color": (230, 74, 86)},
    2: {"name": "ED", "color": (45, 179, 126)},
    4: {"name": "ET", "color": (245, 184, 64)},
}


@dataclass(frozen=True)
class SpatialTransform:
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    resampled_shape: tuple[int, int, int]
    target_shape: tuple[int, int, int]
    source_bounds: tuple[tuple[int, int], ...]
    target_bounds: tuple[tuple[int, int], ...]

    @property
    def source_slices(self):
        return tuple(slice(start, stop) for start, stop in self.source_bounds)

    @property
    def target_slices(self):
        return tuple(slice(start, stop) for start, stop in self.target_bounds)


@dataclass
class PreprocessResult:
    tensor: np.ndarray
    transform: SpatialTransform
    brain_mask: np.ndarray


@dataclass
class CaseData:
    case_id: str
    case_name: str
    paths: dict[str, Path]
    volumes: dict[str, np.ndarray]
    images: dict[str, nib.Nifti1Image]
    prediction_path: Path | None = None
    prediction: np.ndarray | None = None
    prediction_model_space: np.ndarray | None = None
    brain_mask: np.ndarray | None = None
    brain_mask_model_space: np.ndarray | None = None
    transform: SpatialTransform | None = None
    inference_result: dict | None = None
    mesh_result: dict | None = None
    model_id: str | None = None
    model_name: str | None = None

    @property
    def shape(self):
        return tuple(int(value) for value in self.volumes["t1ce"].shape)

    @property
    def spacing(self):
        return tuple(float(value) for value in self.images["t1ce"].header.get_zooms()[:3])


class ModelService:
    def __init__(
        self,
        checkpoint_path: Path,
        model_id: str | None = None,
        model_name: str | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_id = model_id or checkpoint_path.stem
        self.model_name = model_name or self.model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.checkpoint_config = {}
        self.region_thresholds = (0.5, 0.5, 0.5)
        self._lock = threading.RLock()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True, warn_only=True)

    @property
    def loaded(self):
        return self.model is not None

    def load(self):
        with self._lock:
            if self.model is not None:
                return self.model
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Khong tim thay checkpoint: {self.checkpoint_path}")

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint khong chua state_dict hop le.")

            if "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            else:
                state = checkpoint
            if not isinstance(state, dict):
                raise ValueError("Checkpoint khong chua model_state_dict hop le.")

            config = checkpoint.get("config", {})
            if config and not isinstance(config, dict):
                raise ValueError("Config trong checkpoint khong hop le.")
            validate_checkpoint_contract(config)

            normalized_state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
            validate_state_dict_contract(normalized_state)
            model = LATUPNet(
                in_channels=MODEL_INPUT_CHANNELS,
                num_classes=MODEL_OUTPUT_REGIONS,
                use_se=True,
                dropout_rate=0.2,
            )
            model.load_state_dict(normalized_state, strict=True)
            model.to(self.device).eval()
            self.checkpoint_config = config
            self.region_thresholds = (
                float(config.get("wt_threshold", 0.5)),
                float(config.get("tc_threshold", 0.5)),
                float(config.get("et_threshold", 0.5)),
            )
            self.model = model
            return model

    def diagnose(self, case: CaseData):
        with self._lock:
            started = time.perf_counter()
            model = self.load()
            prepared = preprocess(case.volumes, case.spacing)
            tensor = torch.from_numpy(prepared.tensor).unsqueeze(0).to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            with torch.inference_mode():
                logits = model(tensor)
                probabilities = torch.sigmoid(logits)
                prediction = region_probabilities_to_brats_labels(
                    probabilities,
                    thresholds=self.region_thresholds,
                ).squeeze(0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            prediction_model = prediction.cpu().numpy().astype(np.uint8)
            prediction_model[~prepared.brain_mask] = 0
            prediction_native = inverse_crop_pad_resample(
                prediction_model,
                prepared.transform,
                order=0,
            ).astype(np.uint8, copy=False)

            brain_mask_native = build_brain_mask(case.volumes)
            prediction_native[~brain_mask_native] = 0

            case.transform = prepared.transform
            case.brain_mask_model_space = prepared.brain_mask
            case.brain_mask = brain_mask_native
            case.prediction_model_space = prediction_model
            case.prediction = prediction_native
            case.mesh_result = None
            case.model_id = self.model_id
            case.model_name = self.model_name
            case.prediction_path = (
                case.paths["t1ce"].parent
                / f"prediction_{self.model_id}.nii.gz"
            )

            reference = case.images["t1ce"]
            output_header = reference.header.copy()
            output_header.set_data_dtype(np.uint8)
            nib.save(
                nib.Nifti1Image(prediction_native, reference.affine, output_header),
                str(case.prediction_path),
            )

            probabilities_np = probabilities.squeeze(0).cpu().numpy()
            elapsed = time.perf_counter() - started
            result = build_result(
                case,
                probabilities_np,
                elapsed,
                model_id=self.model_id,
                model_name=self.model_name,
            )
            case.inference_result = result
            return result


def validate_checkpoint_contract(config):
    if not config:
        return
    expected = {
        "input_modalities": ["t1ce", "t2", "flair"],
        "patch_size": TARGET_SHAPE,
        "target_spacing": TARGET_SPACING,
        "normalization": "minmax",
        "use_percentile_clip": True,
        "wavelet_sources": [
            "t1ce",
            "t1ce_minus_flair",
            "t2_minus_t1ce",
        ],
        "num_mri_channels": 3,
        "num_wavelet_channels": 6,
        "input_channels": MODEL_INPUT_CHANNELS,
        "num_classes": MODEL_OUTPUT_REGIONS,
        "output_representation": "WT_TC_ET_regions",
        "output_activation": "sigmoid",
        "wavelet_name": WAVELET_NAME,
        "wavelet_level": WAVELET_LEVEL,
        "wavelet_mode": WAVELET_MODE,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key, value) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (mong doi {expected_value!r})"
            for key, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"Checkpoint khong dung contract Wavelet LATUPNet: {details}")


def validate_state_dict_contract(state):
    stem = state.get("pc_block.shared_conv.weight")
    head = state.get("final_conv.weight")
    if stem is None or head is None:
        raise ValueError("Checkpoint thieu trong so stem hoac output head.")
    if stem.ndim != 5 or int(stem.shape[1]) != MODEL_INPUT_CHANNELS:
        raise ValueError(
            f"Checkpoint phai nhan {MODEL_INPUT_CHANNELS} kenh, "
            f"nhan duoc shape {tuple(stem.shape)}."
        )
    if head.ndim != 5 or int(head.shape[0]) != MODEL_OUTPUT_REGIONS:
        raise ValueError(
            f"Checkpoint phai tra {MODEL_OUTPUT_REGIONS} vung, "
            f"nhan duoc shape {tuple(head.shape)}."
        )


def region_probabilities_to_brats_labels(probabilities, thresholds=(0.5, 0.5, 0.5)):
    if probabilities.ndim != 5 or probabilities.shape[1] != MODEL_OUTPUT_REGIONS:
        raise ValueError(
            "Probabilities phai co shape (B, 3, D, H, W) cho WT, TC va ET."
        )
    wt_threshold, tc_threshold, et_threshold = thresholds
    pred_wt = probabilities[:, 0] >= wt_threshold
    pred_tc = probabilities[:, 1] >= tc_threshold
    pred_et = probabilities[:, 2] >= et_threshold

    pred_tc = pred_tc | pred_et
    pred_wt = pred_wt | pred_tc

    labels = torch.zeros_like(pred_wt, dtype=torch.int64)
    labels[pred_wt] = 2
    labels[pred_tc] = 1
    labels[pred_et] = 4
    return labels


def load_case(case_id: str, case_name: str, paths: dict[str, Path]):
    images = {}
    volumes = {}
    for modality, path in paths.items():
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"{modality.upper()} phai la volume 3D, nhan duoc shape {image.shape}.")
        images[modality] = image
        volumes[modality] = image.get_fdata(dtype=np.float32)

    shapes = {volume.shape for volume in volumes.values()}
    if len(shapes) != 1:
        raise ValueError("Ba modality phai co cung kich thuoc volume.")

    reference = images["t1ce"]
    reference_spacing = reference.header.get_zooms()[:3]
    for modality in ("t2", "flair"):
        if not np.allclose(images[modality].affine, reference.affine, rtol=1e-5, atol=1e-3):
            raise ValueError(f"Affine cua {modality.upper()} khong trung voi T1CE.")
        spacing = images[modality].header.get_zooms()[:3]
        if not np.allclose(spacing, reference_spacing, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Voxel spacing cua {modality.upper()} khong trung voi T1CE.")

    return CaseData(case_id, case_name, paths, volumes, images)


def create_spatial_transform(original_shape, spacing, target_shape=TARGET_SHAPE):
    original_shape = tuple(int(value) for value in original_shape)
    spacing = tuple(float(value) for value in spacing)
    if np.allclose(spacing, TARGET_SPACING, rtol=1e-3, atol=1e-3):
        resampled_shape = original_shape
    else:
        resampled_shape = tuple(
            max(1, int(round(size * source_spacing / target_spacing)))
            for size, source_spacing, target_spacing in zip(original_shape, spacing, TARGET_SPACING)
        )

    source_bounds = []
    target_bounds = []
    for current, target in zip(resampled_shape, target_shape):
        copy_size = min(current, target)
        source_start = max(0, (current - copy_size) // 2)
        target_start = max(0, (target - copy_size) // 2)
        source_bounds.append((source_start, source_start + copy_size))
        target_bounds.append((target_start, target_start + copy_size))
    return SpatialTransform(
        original_shape=original_shape,
        original_spacing=spacing,
        resampled_shape=resampled_shape,
        target_shape=tuple(int(value) for value in target_shape),
        source_bounds=tuple(source_bounds),
        target_bounds=tuple(target_bounds),
    )


def preprocess(volumes: dict[str, np.ndarray], spacing):
    shape = tuple(int(value) for value in volumes["t1ce"].shape)
    if any(tuple(volume.shape) != shape for volume in volumes.values()):
        raise ValueError("Ba modality phai co cung kich thuoc truoc preprocessing.")

    transform = create_spatial_transform(shape, spacing)
    cropped_raw = {}
    for modality in ("t1ce", "t2", "flair"):
        volume = np.nan_to_num(volumes[modality], nan=0.0, posinf=0.0, neginf=0.0)
        resampled = resample_to_shape(volume, transform.resampled_shape, order=1)
        cropped_raw[modality] = apply_crop_pad(resampled, transform)

    brain_mask = np.logical_or.reduce(
        [cropped_raw[modality] > 0 for modality in ("t1ce", "t2", "flair")]
    )
    normalized = {
        modality: normalize_in_brain(cropped_raw[modality], brain_mask)
        for modality in ("t1ce", "t2", "flair")
    }

    t1ce = normalized["t1ce"]
    t2 = normalized["t2"]
    flair = normalized["flair"]
    wavelet_sources = (t1ce, t1ce - flair, t2 - t1ce)
    wavelet_channels = []
    for source in wavelet_sources:
        wavelet_channels.extend(wavelet_energy_maps(source, brain_mask))

    tensor = np.stack(
        [t1ce, t2, flair, *wavelet_channels],
        axis=0,
    ).astype(np.float32, copy=False)
    tensor[:, ~brain_mask] = 0.0
    if tensor.shape != (MODEL_INPUT_CHANNELS, *TARGET_SHAPE):
        raise RuntimeError(f"Tensor model co shape khong hop le: {tensor.shape}.")

    return PreprocessResult(
        tensor=tensor,
        transform=transform,
        brain_mask=brain_mask,
    )


def normalize_in_brain(volume, brain_mask, use_percentile_clip=True):
    volume = np.asarray(volume, dtype=np.float32)
    normalized = np.zeros(volume.shape, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool) & np.isfinite(volume)
    if not mask.any():
        return normalized

    working = volume.copy()
    values = working[mask]
    if use_percentile_clip:
        low, high = np.percentile(values, PERCENTILE_RANGE)
        working = np.clip(working, low, high)
        values = working[mask]

    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum >= 1e-6:
        normalized[mask] = (working[mask] - minimum) / (maximum - minimum)
    normalized[~brain_mask] = 0.0
    return normalized


def normalize_minmax_volume(volume):
    return normalize_in_brain(
        volume,
        np.isfinite(volume) & (volume > 0),
    )


def resize_wavelet_map(volume):
    if tuple(volume.shape) == TARGET_SHAPE:
        return volume.astype(np.float32, copy=False)
    return sk_resize(
        volume,
        output_shape=TARGET_SHAPE,
        order=1,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)


def wavelet_energy_maps(source, brain_mask):
    coefficients = pywt.wavedecn(
        source.astype(np.float32),
        wavelet=WAVELET_NAME,
        mode=WAVELET_MODE,
        level=WAVELET_LEVEL,
        axes=(0, 1, 2),
    )
    maps = []
    for detail in reversed(coefficients[1:]):
        energy = None
        for subband in detail.values():
            component = np.square(np.abs(subband).astype(np.float32))
            energy = component if energy is None else energy + component
        energy = np.sqrt(energy + 1e-8)
        energy = resize_wavelet_map(energy)
        maps.append(normalize_in_brain(energy, brain_mask))
    if len(maps) != WAVELET_LEVEL:
        raise RuntimeError(
            f"Wavelet phai tao {WAVELET_LEVEL} energy maps, nhan duoc {len(maps)}."
        )
    return maps


def build_brain_mask(volumes):
    return np.logical_or.reduce(
        [np.isfinite(volume) & (volume > 0) for volume in volumes.values()]
    )


def resample_to_shape(volume, output_shape, order):
    output_shape = tuple(int(value) for value in output_shape)
    if tuple(volume.shape) == output_shape:
        return volume.copy()
    factors = tuple(target / current for target, current in zip(output_shape, volume.shape))
    return fit_shape(nd_zoom(volume, factors, order=order), output_shape)


def apply_crop_pad(volume, transform: SpatialTransform):
    result = np.zeros(transform.target_shape, dtype=volume.dtype)
    result[transform.target_slices] = volume[transform.source_slices]
    return result


def inverse_crop_pad_resample(volume, transform: SpatialTransform, order=0):
    if tuple(volume.shape) != transform.target_shape:
        raise ValueError(
            f"Model volume co shape {volume.shape}, mong doi {transform.target_shape}."
        )
    resampled = np.zeros(transform.resampled_shape, dtype=volume.dtype)
    resampled[transform.source_slices] = volume[transform.target_slices]
    return resample_to_shape(resampled, transform.original_shape, order=order)


def fit_shape(volume, target_shape):
    result = np.zeros(target_shape, dtype=volume.dtype)
    source_slices = []
    target_slices = []
    for current, target in zip(volume.shape, target_shape):
        copy_size = min(current, target)
        source_start = max(0, (current - copy_size) // 2)
        target_start = max(0, (target - copy_size) // 2)
        source_slices.append(slice(source_start, source_start + copy_size))
        target_slices.append(slice(target_start, target_start + copy_size))
    result[tuple(target_slices)] = volume[tuple(source_slices)]
    return result


def build_preview(case: CaseData, index=None):
    depth = case.shape[2]
    index = depth // 2 if index is None else int(np.clip(index, 0, depth - 1))
    return {
        "case_id": case.case_id,
        "case_name": case.case_name,
        "shape": list(case.shape),
        "spacing": [round(value, 3) for value in case.spacing],
        "slice_index": index,
        "slice_max": depth - 1,
        "slices": {
            modality: volume_slice_data_url(case.volumes[modality], "axial", index)
            for modality in ("t1ce", "t2", "flair")
        },
    }


def build_result(case, probabilities, elapsed, model_id=None, model_name=None):
    prediction = case.prediction
    voxel_volume_ml = float(np.prod(case.spacing)) / 1000.0
    class_stats = []
    for label, metadata in LABELS.items():
        count = int(np.count_nonzero(prediction == label))
        class_stats.append(
            {
                "label": label,
                "name": metadata["name"],
                "voxels": count,
                "volume_ml": round(count * voxel_volume_ml, 3),
                "color": color_hex(metadata["color"]),
            }
        )

    tumor_voxels = int(np.count_nonzero(prediction))
    brain_voxels = int(np.count_nonzero(case.brain_mask))
    outside_brain_voxels = int(np.count_nonzero(prediction[~case.brain_mask]))
    tumor_to_brain_ratio = tumor_voxels / brain_voxels if brain_voxels else 0.0
    model_mask = case.prediction_model_space > 0
    wt_confidence = probabilities[0]
    mean_confidence = (
        float(wt_confidence[model_mask].mean())
        if model_mask.any()
        else 0.0
    )
    default_slices = {
        axis: build_result_slice(case, axis, best_tumor_slice(prediction, axis))
        for axis in AXES
    }
    quality_warning = tumor_to_brain_ratio > 0.25

    return {
        "case_id": case.case_id,
        "case_name": case.case_name,
        "model_id": model_id or case.model_id,
        "model_name": model_name or case.model_name,
        "inference_seconds": round(elapsed, 2),
        "mean_confidence": round(mean_confidence * 100, 1),
        "confidence_basis": "mean WT sigmoid probability over predicted tumor voxels",
        "model_input_channels": MODEL_INPUT_CHANNELS,
        "output_regions": ["WT", "TC", "ET"],
        "tumor_volume_ml": round(tumor_voxels * voxel_volume_ml, 3),
        "tumor_voxels": tumor_voxels,
        "brain_voxels": brain_voxels,
        "tumor_to_brain_ratio": round(tumor_to_brain_ratio, 6),
        "outside_brain_voxels": outside_brain_voxels,
        "quality_warning": quality_warning,
        "quality_message": (
            "Tỷ lệ vùng dự đoán so với não cao bất thường. Cần kiểm tra lại input và kết quả."
            if quality_warning
            else "Prediction nằm trong brain mask và tỷ lệ thể tích ở mức hợp lý."
        ),
        "class_stats": class_stats,
        "default_slices": default_slices,
        "download_url": f"/api/cases/{case.case_id}/prediction",
        "mesh_url": f"/api/cases/{case.case_id}/mesh",
    }


def best_tumor_slice(mask, axis):
    axis_index = AXES[axis]
    reduce_axes = tuple(index for index in range(3) if index != axis_index)
    counts = np.count_nonzero(mask, axis=reduce_axes)
    return int(counts.argmax()) if counts.max() else mask.shape[axis_index] // 2


def build_result_slice(case: CaseData, axis, index):
    if case.prediction is None:
        raise ValueError("Ca nay chua co ket qua du doan.")
    if axis not in AXES:
        raise ValueError("axis phai la axial, coronal hoac sagittal.")
    axis_index = AXES[axis]
    maximum = case.shape[axis_index] - 1
    index = int(np.clip(index, 0, maximum))
    return {
        "axis": axis,
        "index": index,
        "slice_max": maximum,
        "image": overlay_data_url(case.volumes["flair"], case.prediction, axis, index),
    }


def normalize_uint8(image):
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    positive = finite[finite > 0]
    values = positive if positive.size else finite
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = np.clip((image - low) / (high - low), 0, 1)
    return (normalized * 255).astype(np.uint8)


def extract_slice(volume, axis, index):
    axis_index = AXES[axis]
    slices = [slice(None)] * 3
    slices[axis_index] = index
    return np.rot90(volume[tuple(slices)])


def volume_slice_data_url(volume, axis, index):
    image = extract_slice(volume, axis, index)
    return image_data_url(Image.fromarray(normalize_uint8(image)))


def overlay_data_url(volume, mask, axis, index):
    base = normalize_uint8(extract_slice(volume, axis, index))
    rgb = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    mask_slice = extract_slice(mask, axis, index)
    for label, metadata in LABELS.items():
        selected = mask_slice == label
        if selected.any():
            color = np.asarray(metadata["color"], dtype=np.float32)
            rgb[selected] = rgb[selected] * 0.35 + color * 0.65
    return image_data_url(Image.fromarray(rgb.astype(np.uint8)))


def image_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_meshes(case: CaseData):
    if case.prediction is None or case.brain_mask is None:
        raise ValueError("Ca nay chua co ket qua du doan.")
    if case.mesh_result is not None:
        return case.mesh_result

    meshes = [
        build_single_mesh(
            case.brain_mask,
            case.shape,
            case.spacing,
            label="brain",
            name="Brain",
            color="#9aa7a1",
            opacity=0.16,
            step_size=4,
        )
    ]
    for label, metadata in LABELS.items():
        meshes.append(
            build_single_mesh(
                case.prediction == label,
                case.shape,
                case.spacing,
                label=label,
                name=metadata["name"],
                color=color_hex(metadata["color"]),
                opacity=0.82,
                step_size=1,
            )
        )
    case.mesh_result = {
        "case_id": case.case_id,
        "model_id": case.model_id,
        "model_name": case.model_name,
        "meshes": meshes,
    }
    return case.mesh_result


def build_single_mesh(binary, shape, spacing, label, name, color, opacity, step_size):
    empty = {
        "label": label,
        "name": name,
        "color": color,
        "opacity": opacity,
        "vertices": [],
        "faces": [],
    }
    if np.count_nonzero(binary) < 8:
        return empty
    try:
        vertices, faces, _, _ = marching_cubes(
            binary.astype(np.uint8),
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )
        physical_shape = np.asarray(shape, dtype=np.float32) * np.asarray(spacing, dtype=np.float32)
        center = (np.asarray(shape, dtype=np.float32) - 1.0) * np.asarray(spacing, dtype=np.float32) / 2.0
        scale = max(float(physical_shape.max()) / 2.0, 1.0)
        vertices = (vertices - center) / scale
        vertices = vertices[:, [0, 2, 1]]
        return {
            **empty,
            "vertices": np.round(vertices, 4).reshape(-1).tolist(),
            "faces": faces.astype(np.int32).reshape(-1).tolist(),
        }
    except (RuntimeError, ValueError):
        return empty


def color_hex(color):
    return "#%02x%02x%02x" % color
