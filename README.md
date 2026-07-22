# TumorMesh Studio

TumorMesh Studio is a local web application for segmenting brain tumors from three MRI volumes and viewing the result in 2D and 3D.

It accepts BraTS-style `T1ce`, `T2`, and `FLAIR` NIfTI files, applies the same wavelet preprocessing used during training, predicts tumor labels, and returns:

- FLAIR overlays in axial, coronal, and sagittal views;
- a downloadable NIfTI segmentation mask;
- tumor volume and basic quality checks;
- an interactive 3D brain and tumor reconstruction.

> Research use only. This project is for technical demonstration and research. It is not a medical device and must not be used as a clinical diagnosis tool.

## What You Need

- Windows, Linux, or macOS;
- Python 3.10 or 3.11;
- 8 GB RAM minimum, 16 GB recommended;
- an NVIDIA GPU is recommended. CPU inference is supported but considerably slower;
- three MRI files from the same patient: `T1ce`, `T2`, and `FLAIR`.

The input is expected to follow the BraTS convention: 3D skull-stripped MRI volumes, already registered to the same space, with background outside the brain equal to `0`.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/GauKho/3D-Brain-Tumor-Reconstruction-and-Segmentation.git
cd 3D-Brain-Tumor-Reconstruction-and-Segmentation
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the following once in the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### 3. Confirm the two model files exist

```text
outputs/best_latupnet_wavelet_region3_sigmoid_priority1.pth
REL/working/working/best_model_rl_rel_ppo.pth
```

### 4. Run the Studio

Windows:

```powershell
.\run_app.ps1
```

The script finds an available port between `8000` and `8010`, then prints the URL. Open that URL in a browser.

On any operating system, you can also start FastAPI directly:

```bash
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Do not open `webapp/static/index.html` directly because the UI needs the FastAPI backend.

## Using the Application

1. Select one NIfTI file for each modality: `T1ce`, `T2`, and `FLAIR`.
2. Wait for the input preview and use the axial slider to check the three images.
3. Open **Mô hình phân đoạn** and choose a model.
4. Click **Tiến hành chẩn đoán**.
5. Review the model name, the 2D overlays, tumor statistics, and the 3D reconstruction.
6. Download the NIfTI mask if required.

You can run the same uploaded case with the other model. The most recent inference becomes the active result shown in 2D and 3D.

## Available Models

| Name in Studio | File | Meaning |
|---|---|---|
| `Latup-net-wavelet` | `outputs/best_latupnet_wavelet_region3_sigmoid_priority1.pth` | LATUPNet with nine input channels, 3D wavelet features, and fixed regional loss weights. |
| `Rel_ppo` | `REL/working/working/best_model_rl_rel_ppo.pth` | LATUPNet weights produced during the PPO joint-training experiment. |

Both models use the same inference contract:

```text
Input:  T1ce + T2 + FLAIR + 6 wavelet energy maps
Tensor: 9 x 128 x 128 x 128
Output: WT, TC, ET sigmoid maps -> BraTS labels {0, 1, 2, 4}
```

`Rel_ppo` selects the LATUPNet weights saved from the PPO experiment. The PPO policy itself is not run inside the Studio during inference.

## Input Checks

The backend rejects the input if any of the following is invalid:

- a file is not a 3D NIfTI volume;
- the three volumes have different shapes;
- T2 or FLAIR has a different affine matrix from T1ce;
- T2 or FLAIR has different voxel spacing from T1ce.

The file names do not need to follow a fixed pattern. Choose the correct file in the corresponding modality field.

## Processing Pipeline

```text
T1ce + T2 + FLAIR NIfTI
        |
Geometry validation
        |
Resample to 1 mm + center crop/pad to 128 cubed
        |
P1-P99 clipping + masked Min-Max normalization
        |
Six 3D Symlet-8 wavelet energy maps
        |
Nine-channel LATUPNet inference
        |
WT / TC / ET hierarchy -> labels 0, 1, 2, 4
        |
Native-space NIfTI + 2D overlays + Marching Cubes 3D mesh
```

The prediction is mapped back to the original image shape. The output NIfTI preserves the T1ce affine and header, so it can be opened in a medical image viewer in the correct patient space.

## Output Labels

| Value | Region | Studio color |
|---:|---|---|
| `0` | Background / non-tumor | Transparent |
| `1` | NCR/NET: necrotic and non-enhancing tumor core | Red |
| `2` | ED: peritumoral edema | Green |
| `4` | ET: enhancing tumor | Yellow |

BraTS evaluation regions are derived as follows:

```text
WT = labels 1 + 2 + 4
TC = labels 1 + 4
ET = label 4
```

## 2D and 3D Views

The Studio first displays 2D FLAIR overlays so results are available quickly. It then requests mesh data separately for the 3D view.

The 3D reconstruction uses Marching Cubes:

- a gray, semi-transparent brain surface provides context;
- NCR/NET, ED, and ET are generated as separate surfaces;
- brain mesh uses a lower detail step to reduce payload;
- tumor meshes retain higher detail;
- Three.js provides rotation, zoom, layer visibility, opacity, wireframe, auto-rotate, and camera reset.

The mesh is a visualization of the predicted voxel mask. It does not create a second prediction and does not improve the segmentation result.

## Files Created After Inference

Each uploaded case is stored temporarily under `runtime/cases/`. The active prediction is saved as:

```text
pred_<patient>_<model_id>.nii.gz
```

For example:

```text
pred_BraTS20_Validation_031_latup-net-wavelet.nii.gz
pred_BraTS20_Validation_031_rel-ppo.nii.gz
```

## REST API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/status` | Available models and service status |
| `POST` | `/api/cases` | Upload multipart `t1ce`, `t2`, and `flair` |
| `POST` | `/api/cases/{id}/diagnose?model_id=<id>` | Run selected model |
| `GET` | `/api/cases/{id}/slice?index=n` | Input MRI preview |
| `GET` | `/api/cases/{id}/result-slice?axis=axial&index=n` | Segmentation overlay |
| `GET` | `/api/cases/{id}/mesh` | 3D mesh data |
| `GET` | `/api/cases/{id}/prediction` | Download NIfTI mask |

Valid model IDs are `latup-net-wavelet` and `rel-ppo`.

## Run Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the preprocessing contract, crop/pad inversion, nine-channel tensor construction, model loading, model routing, output labels, and frontend model-picker wiring.

## Troubleshooting

### Port already in use

Use the provided Windows launcher:

```powershell
.\run_app.ps1
```

Or choose a different port manually:

```bash
python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8001
```

### Studio says a model is missing

Confirm both files from the **Quick Start** section exist. The model menu disables a model whose file is not available.

### 3D panel reports `Cannot read properties of null (reading 'precision')`

This is a browser WebGL context issue, not a NIfTI or segmentation error. The Studio now suspends the old 3D scene before new inference, clears its GPU buffers, and rebuilds the canvas if the context is lost. Press `Ctrl+F5` after updating. If it persists, close older Studio tabs and enable Hardware Acceleration in the browser.

### GPU out of memory

Close other GPU-heavy programs and run one inference at a time. The model always receives `128 x 128 x 128` volumes; changing this size breaks the training and inference contract. CPU inference remains possible but is slower.

### Input is rejected

Make sure T1ce, T2, and FLAIR are from the same patient and have identical shape, affine, and voxel spacing. Do not resize the modalities independently.

## Project Layout

```text
webapp/
  app.py              FastAPI routes and model registry
  inference.py        preprocessing, inference, NIfTI, overlays, meshes
  model.py            LATUPNet architecture
  static/             Studio frontend and local Three.js assets
outputs/              Latup-net-wavelet model file
REL/working/working/  Rel_ppo model file
tests/                unit and integration-style tests
run_app.ps1           Windows launcher
requirements.txt      Python dependencies
```

## Technology

- AI and medical imaging: PyTorch, NumPy, NiBabel, SciPy, scikit-image, PyWavelets, Pillow.
- Backend: FastAPI and Uvicorn.
- Frontend: HTML, CSS, JavaScript modules, Three.js, OrbitControls, WebGL.
- Medical format: NIfTI (`.nii`, `.nii.gz`) with affine and voxel-spacing metadata.
