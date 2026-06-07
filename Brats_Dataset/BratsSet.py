from torch.utils.data import Dataset
from pathlib import Path
from .pre_processing import load_volume, resample_volume, crop_or_pad_3d
import numpy as np
import torch
from scipy.ndimage import zoom as nd_zoom

class BraTSDataset3D(Dataset):
    def __init__(self, patient_dirs, target_spacing=(1, 1, 1), target_size=(128, 128, 128)):
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

            # 1. Resample về target_spacing (1, 1, 1) theo lý thuyết
            if not np.allclose(original_spacing, self.target_spacing, rtol=1e-3):
                resampled_volumes = {
                    mod: resample_volume(volumes[mod], original_spacing, self.target_spacing, order=1)
                    for mod in self.modalities
                }
                resampled_seg = resample_volume(seg, original_spacing, self.target_spacing, order=0)
            else:
                resampled_volumes = volumes
                resampled_seg     = seg

            # 2. Cắt hoặc Pad về target_size (128, 128, 128)
            cropped_volumes = {
                mod: crop_or_pad_3d(resampled_volumes[mod], self.target_size)
                for mod in self.modalities
            }
            cropped_seg = crop_or_pad_3d(resampled_seg, self.target_size)

            # 3. Chuẩn hóa Min-Max chỉ dựa trên vùng não (Voxel > 0)
            for modality in self.modalities:
                vol = cropped_volumes[modality]
                mask_brain = vol > 0  # Loại bỏ vùng nền đen
                
                if np.any(mask_brain):
                    vmin, vmax = vol[mask_brain].min(), vol[mask_brain].max()
                    # Chuẩn hóa toàn bộ khối dựa trên min/max của vùng não
                    if vmax > vmin:
                        vol_norm = (vol - vmin) / (vmax - vmin)
                        # Đảm bảo giữ nguyên vùng nền bên ngoài bằng 0
                        vol_norm[~mask_brain] = 0
                        cropped_volumes[modality] = vol_norm
                    else:
                        cropped_volumes[modality] = np.zeros_like(vol)
                else:
                    cropped_volumes[modality] = np.zeros_like(vol)

            # Chuyển đổi sang Tensor PyTorch
            image_np = np.stack([cropped_volumes[mod] for mod in self.modalities], axis=0).astype(np.float32)
            mask_np  = cropped_seg.astype(np.int64)

            return torch.from_numpy(image_np), torch.from_numpy(mask_np), patient_id

        except Exception as e:
            print(f"[WARN] Lỗi khi xử lý {patient_id}: {e}")
            return torch.zeros((3, *self.target_size)), torch.zeros(self.target_size).long(), patient_id