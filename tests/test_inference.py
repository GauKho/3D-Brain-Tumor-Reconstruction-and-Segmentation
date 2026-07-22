import base64
import io
import json
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image

from webapp.app import (
    DEFAULT_MODEL_ID,
    MODEL_SPECS,
    cases,
    cases_lock,
    diagnose,
    model_services,
    register_case,
    status,
)
from webapp.inference import (
    ModelService,
    apply_crop_pad,
    build_result_slice,
    create_spatial_transform,
    inverse_crop_pad_resample,
    load_case,
    normalize_minmax_volume,
    preprocess,
    region_probabilities_to_brats_labels,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "BraTS20_Validation_031_t1ce.nii"
CHECKPOINT = ROOT / "outputs" / "best_latupnet_wavelet_region3_sigmoid_priority1.pth"
REL_CHECKPOINT = ROOT / "REL" / "working" / "working" / "best_model_rl_rel_ppo.pth"
TRAINING_NOTEBOOK = ROOT / "latupnet-wavelet-preprocessing-for-brats.ipynb"


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, _tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])


class SpatialTransformTests(unittest.TestCase):
    def test_center_pad_and_inverse_preserve_center_and_edges(self):
        shape = (64, 80, 96)
        volume = np.zeros(shape, dtype=np.uint8)
        markers = ((0, 0, 0), (32, 40, 48), (63, 79, 95))
        for value, position in enumerate(markers, start=1):
            volume[position] = value

        transform = create_spatial_transform(shape, (1, 1, 1))
        model_volume = apply_crop_pad(volume, transform)
        restored = inverse_crop_pad_resample(model_volume, transform)

        np.testing.assert_array_equal(restored, volume)
        self.assertEqual(model_volume.shape, (128, 128, 128))

    def test_center_crop_and_inverse_keep_retained_boundary_markers(self):
        shape = (160, 150, 140)
        transform = create_spatial_transform(shape, (1, 1, 1))
        volume = np.zeros(shape, dtype=np.uint8)
        starts = tuple(bounds[0] for bounds in transform.source_bounds)
        stops = tuple(bounds[1] - 1 for bounds in transform.source_bounds)
        center = tuple(size // 2 for size in shape)
        volume[starts] = 1
        volume[center] = 2
        volume[stops] = 3

        restored = inverse_crop_pad_resample(apply_crop_pad(volume, transform), transform)

        self.assertEqual(int(restored[starts]), 1)
        self.assertEqual(int(restored[center]), 2)
        self.assertEqual(int(restored[stops]), 3)
        self.assertEqual(int(restored[0, 0, 0]), 0)

    def test_normalization_keeps_background_zero(self):
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        volume[1, 1, 1] = 10
        volume[2, 2, 2] = 20

        normalized = normalize_minmax_volume(volume)

        self.assertEqual(float(normalized[0, 0, 0]), 0.0)
        self.assertEqual(float(normalized[1, 1, 1]), 0.0)
        self.assertEqual(float(normalized[2, 2, 2]), 1.0)

    def test_preprocess_builds_nine_channels_and_preserves_background(self):
        shape = (40, 44, 48)
        x, y, z = np.indices(shape, dtype=np.float32)
        support = np.zeros(shape, dtype=bool)
        support[8:32, 9:35, 10:38] = True
        t1ce = np.zeros(shape, dtype=np.float32)
        t2 = np.zeros(shape, dtype=np.float32)
        flair = np.zeros(shape, dtype=np.float32)
        t1ce[support] = x[support] + 0.2 * y[support]
        t2[support] = y[support] + 0.3 * z[support]
        flair[support] = z[support] + 0.4 * x[support]
        volumes = {
            "t1ce": t1ce,
            "t2": t2,
            "flair": flair,
        }

        result = preprocess(volumes, spacing=(1.0, 1.0, 1.0))

        self.assertEqual(result.tensor.shape, (9, 128, 128, 128))
        self.assertTrue(np.isfinite(result.tensor).all())
        self.assertFalse(np.any(result.tensor[:, ~result.brain_mask]))
        self.assertGreaterEqual(float(result.tensor.min()), 0.0)
        self.assertLessEqual(float(result.tensor.max()), 1.0)
        self.assertTrue(np.all(np.count_nonzero(result.tensor[3:], axis=(1, 2, 3)) > 0))

    def test_region_predictions_preserve_et_inside_tc_inside_wt(self):
        probabilities = torch.zeros((1, 3, 2, 2, 2), dtype=torch.float32)
        probabilities[0, 0, 0, 0, 0] = 0.9
        probabilities[0, 1, 0, 0, 1] = 0.9
        probabilities[0, 2, 0, 1, 0] = 0.9

        labels = region_probabilities_to_brats_labels(probabilities)[0].numpy()

        self.assertEqual(int(labels[0, 0, 0]), 2)
        self.assertEqual(int(labels[0, 0, 1]), 1)
        self.assertEqual(int(labels[0, 1, 0]), 4)
        self.assertEqual(set(np.unique(labels).tolist()), {0, 1, 2, 4})


@unittest.skipUnless(
    CHECKPOINT.exists() and REL_CHECKPOINT.exists(),
    "Missing one or more Studio checkpoints",
)
class CheckpointContractTests(unittest.TestCase):
    def test_both_checkpoints_load_exact_nine_channel_three_region_model(self):
        checkpoints = (
            ("latup-net-wavelet", "Latup-net-wavelet", CHECKPOINT),
            ("rel-ppo", "Rel_ppo", REL_CHECKPOINT),
        )
        for model_id, model_name, checkpoint in checkpoints:
            with self.subTest(model_id=model_id):
                service = ModelService(checkpoint, model_id=model_id, model_name=model_name)
                model = service.load()

                self.assertEqual(service.model_id, model_id)
                self.assertEqual(service.model_name, model_name)
                self.assertEqual(model.pc_block.shared_conv.in_channels, 9)
                self.assertEqual(model.final_conv.out_channels, 3)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    2_993_539,
                )
                self.assertEqual(service.region_thresholds, (0.5, 0.5, 0.5))

    def test_fixed_checkpoint_exposes_training_contract(self):
        service = ModelService(CHECKPOINT)
        service.load()
        self.assertEqual(
            service.checkpoint_config.get("output_representation"),
            "WT_TC_ET_regions",
        )

    def test_rl_checkpoint_uses_verified_runtime_defaults_without_config(self):
        service = ModelService(REL_CHECKPOINT)
        service.load()
        self.assertEqual(service.checkpoint_config, {})
        self.assertEqual(service.region_thresholds, (0.5, 0.5, 0.5))


class ModelRegistryTests(unittest.TestCase):
    def tearDown(self):
        with cases_lock:
            cases.pop("model-routing", None)

    def test_status_lists_both_selectable_models(self):
        payload = status()
        models = {model["id"]: model for model in payload["models"]}

        self.assertEqual(DEFAULT_MODEL_ID, "latup-net-wavelet")
        self.assertEqual(list(MODEL_SPECS), ["latup-net-wavelet", "rel-ppo"])
        self.assertEqual(models["latup-net-wavelet"]["name"], "Latup-net-wavelet")
        self.assertEqual(models["rel-ppo"]["name"], "Rel_ppo")
        self.assertTrue(models["latup-net-wavelet"]["ready"])
        self.assertTrue(models["rel-ppo"]["ready"])

    def test_diagnose_routes_case_to_requested_model(self):
        case = SimpleNamespace(case_id="model-routing")
        register_case(case)
        expected = {"model_id": "rel-ppo", "model_name": "Rel_ppo"}

        with patch.object(model_services["rel-ppo"], "diagnose", return_value=expected) as mocked:
            result = diagnose(case.case_id, model_id="rel-ppo")

        self.assertEqual(result, expected)
        mocked.assert_called_once_with(case)

    def test_diagnose_rejects_unknown_model(self):
        case = SimpleNamespace(case_id="model-routing")
        register_case(case)

        with self.assertRaises(HTTPException) as captured:
            diagnose(case.case_id, model_id="unknown")

        self.assertEqual(captured.exception.status_code, 400)


class FrontendContractTests(unittest.TestCase):
    def test_model_picker_ids_and_api_query_are_wired(self):
        html = (ROOT / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
        javascript = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        collector = IdCollector()
        collector.feed(html)
        html_ids = set(collector.ids)
        static_js_ids = set(re.findall(r'\$\("([^"]+)"\)', javascript))

        self.assertEqual(len(collector.ids), len(html_ids), "HTML contains duplicate IDs")
        self.assertFalse(static_js_ids - html_ids, "JavaScript references missing HTML IDs")
        self.assertIn("modelPickerButton", html_ids)
        self.assertIn("modelMenu", html_ids)
        self.assertIn("resultModelName", html_ids)
        self.assertIn("diagnose?model_id=", javascript)

    def test_viewer_releases_meshes_before_inference_and_recovers_context(self):
        javascript = (ROOT / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
        three_module = (
            ROOT / "webapp" / "static" / "vendor" / "three.module.js"
        ).read_text(encoding="utf-8")

        self.assertIn("state.viewer?.suspend", javascript)
        self.assertIn("replaceViewerCanvas", javascript)
        self.assertIn("preserveDrawingBuffer: false", javascript)
        self.assertIn("format !== null && format.precision > 0", three_module)


@unittest.skipUnless(TRAINING_NOTEBOOK.exists(), "Missing current training notebook")
class NotebookContractTests(unittest.TestCase):
    def test_runtime_contract_is_grounded_in_current_wavelet_notebook(self):
        notebook = json.loads(TRAINING_NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        required_fragments = (
            "USE_PERCENTILE_CLIP = True",
            "P_LOW, P_HIGH = 1.0, 99.0",
            'WAVELET_NAME = "sym8"',
            "WAVELET_LEVEL = 2",
            'WAVELET_MODE = "symmetric"',
            "INPUT_CHANNELS = NUM_MRI_CHANNELS + NUM_WAVELET_CHANNELS",
            "NUM_CLASSES = 3",
            '"output_representation": "WT_TC_ET_regions"',
            '"output_activation": "sigmoid"',
            "pred_tc = pred_tc | pred_et",
            "pred_wt = pred_wt | pred_tc",
        )

        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)


@unittest.skipUnless(CHECKPOINT.exists() and SAMPLE_DIR.is_dir(), "Missing regression assets")
class CheckpointRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths = {
            modality: next(SAMPLE_DIR.glob(f"*_{modality}.nii*"))
            for modality in ("t1ce", "t2", "flair")
        }
        cls.case = load_case("regression", "BraTS20_Validation_031", paths)
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.case.paths["t1ce"] = Path(cls.temp_dir.name) / paths["t1ce"].name
        cls.result = ModelService(
            CHECKPOINT,
            model_id="latup-net-wavelet",
            model_name="Latup-net-wavelet",
        ).diagnose(cls.case)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_model_prediction_is_nonempty_and_inside_brain(self):
        self.assertEqual(self.result["model_id"], "latup-net-wavelet")
        self.assertEqual(self.result["model_name"], "Latup-net-wavelet")
        self.assertGreater(self.result["tumor_voxels"], 0)
        self.assertGreater(self.result["tumor_volume_ml"], 0)
        self.assertEqual(self.result["outside_brain_voxels"], 0)

    def test_output_nifti_preserves_native_geometry_and_labels(self):
        output = nib.load(str(self.case.prediction_path))
        reference = self.case.images["t1ce"]
        prediction = output.get_fdata().astype(np.uint8)

        self.assertEqual(output.shape, reference.shape)
        np.testing.assert_allclose(output.affine, reference.affine)
        np.testing.assert_allclose(output.header.get_zooms()[:3], reference.header.get_zooms()[:3])
        self.assertEqual(set(np.unique(prediction).tolist()), {0, 1, 2, 4})
        self.assertFalse(np.any(prediction[~self.case.brain_mask]))

    def test_three_default_result_slices_are_valid_png_images(self):
        for axis, data in self.result["default_slices"].items():
            self.assertEqual(data["axis"], axis)
            refreshed = build_result_slice(self.case, axis, data["index"])
            raw = base64.b64decode(refreshed["image"].split(",", 1)[1])
            image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)
            self.assertGreater(int(image.max()), int(image.min()))


if __name__ == "__main__":
    unittest.main()
