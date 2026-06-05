from torch.utils.data import Dataset
from pathlib import Path
from .pre_processing import load_volume, resample_volume
import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom

class BraTSDataset3D(Dataset):
    def __init__(self, patient_dirs, target_spacing=(1, 1, 1), target_size=(128, 128, 128)):
        """
        Args:
            patient_dirs (list): Danh sách các Path dẫn thẳng tới thư mục cụ thể của từng bệnh nhân.
                                 Ví dụ: [Path('.../BraTS20_Training_001'), Path('.../BraTS20_Training_002')]
            target_spacing (tuple): Spacing mục tiêu để resample.
            target_size (tuple): Kích thước khối (H, W, D) đầu ra sau khi crop.
        """
        # ĐỔI TỪ ROOT_DIR THÀNH PATIENT_DIRS: Nhận trực tiếp danh sách đã chia nhỏ từ ngoài vào
        self.patient_dirs = [Path(p) for p in patient_dirs] 
        self.target_spacing = target_spacing
        self.target_size = target_size
        self.modalities = ["t1ce", "t2", "flair"]
        
        print(f"Dataset đã khởi tạo thành công với: {len(self.patient_dirs)} bệnh nhân.")

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id  = patient_dir.name

        try:
            volumes = {}
            for modality in self.modalities:
                volumes[modality], _ = load_volume(patient_dir, patient_id, modality)

            seg, seg_img = load_volume(patient_dir, patient_id, "seg")
            seg = seg.astype(np.uint8)

            original_spacing = seg_img.header.get_zooms()[:3]

            # Resample về target_spacing
            if not np.allclose(original_spacing, self.target_spacing, rtol=1e-3):
                resampled_volumes = {
                    mod: resample_volume(volumes[mod], original_spacing, self.target_spacing, order=1)
                    for mod in self.modalities
                }
                resampled_seg = resample_volume(seg, original_spacing, self.target_spacing, order=0)
            else:
                resampled_volumes = volumes
                resampled_seg     = seg

            # Resize toàn bộ volume về target_size bằng zoom — KHÔNG crop
            current_shape = resampled_seg.shape  # e.g. (240, 240, 155)
            zoom_factors  = tuple(t / c for t, c in zip(self.target_size, current_shape))

            resized_volumes = {
                mod: nd_zoom(resampled_volumes[mod], zoom_factors, order=1)
                for mod in self.modalities
            }
            resized_seg = nd_zoom(resampled_seg, zoom_factors, order=0)  # nearest neighbor cho mask

            # Normalize Min-Max từng modality
            for modality in self.modalities:
                vol  = resized_volumes[modality]
                vmin, vmax = vol.min(), vol.max()
                resized_volumes[modality] = (vol - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vol)

            image_np = np.stack([resized_volumes[mod] for mod in self.modalities], axis=0).astype(np.float32)
            mask_np  = resized_seg.astype(np.int64)

            return torch.from_numpy(image_np), torch.from_numpy(mask_np), patient_id

        except Exception as e:
            print(f"[WARN] Lỗi khi xử lý {patient_id}: {e}")
            return torch.zeros((3, *self.target_size)), torch.zeros(self.target_size).long(), patient_id

