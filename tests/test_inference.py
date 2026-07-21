import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image

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
TRAINING_NOTEBOOK = ROOT / "latupnet-wavelet-preprocessing-for-brats.ipynb"


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


@unittest.skipUnless(CHECKPOINT.exists(), "Missing Wavelet LATUPNet checkpoint")
class CheckpointContractTests(unittest.TestCase):
    def test_checkpoint_loads_exact_nine_channel_three_region_model(self):
        service = ModelService(CHECKPOINT)
        model = service.load()

        self.assertEqual(model.pc_block.shared_conv.in_channels, 9)
        self.assertEqual(model.final_conv.out_channels, 3)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 2_993_539)
        self.assertEqual(
            service.checkpoint_config.get("output_representation"),
            "WT_TC_ET_regions",
        )
        self.assertEqual(service.region_thresholds, (0.5, 0.5, 0.5))


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
        cls.result = ModelService(CHECKPOINT).diagnose(cls.case)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_model_prediction_is_nonempty_and_inside_brain(self):
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
