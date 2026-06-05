from pathlib import Path
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom as nd_zoom

def find_volume(patient_dir, patient_id, modality):
    """Find a valid .nii or .nii.gz file for one BraTS modality, handling inconsistent naming."""
    patient_dir = Path(patient_dir)
    
    # 1. Định nghĩa các mẫu từ khóa tìm kiếm cho từng modality (bất kể viết hoa/thường)
    modality_lower = modality.lower()
    
    if modality_lower == "seg":
        # Tìm các file chứa "seg" hoặc "segm" (để xử lý case bệnh nhân 355)
        search_patterns = ["*seg*.nii", "*seg*.nii.gz", "*Segm*.nii", "*Segm*.nii.gz"]
    else:
        # Với t1, t1ce, t2, flair: tìm file chứa đúng từ khóa đó (ví dụ: *_flair.nii)
        search_patterns = [f"*{modality_lower}.nii", f"*{modality_lower}.nii.gz"]

    # 2. Quét qua các mẫu để tìm file thực tế trong thư mục
    for pattern in search_patterns:
        # Chạy glob không phân biệt hoa thường (trên Windows/Linux tùy cấu hình, dùng rglob/glob)
        hits = list(patient_dir.glob(pattern))
        
        # Nếu không thấy, thử tìm phiên bản viết hoa chữ cái đầu (như Flair, T1ce...)
        if not hits:
            hits = list(patient_dir.glob(pattern.capitalize()))
            
        for path in hits:
            # Kiểm tra xem file có hợp lệ (tồn tại và không trống) không
            if path.exists() and path.stat().st_size > 0:
                return path

    # 3. Nếu không tìm thấy bằng glob, quay lại fallback cơ bản (để hiển thị danh sách đã thử)
    candidates = [
        patient_dir / f"{patient_id}_{modality}.nii",
        patient_dir / f"{patient_id}_{modality}.nii.gz",
    ]
    tried = [str(p) for p in candidates] + [str(patient_dir / p) for p in search_patterns]
    
    zip_candidates = [patient_dir / f"{patient_id}_{modality}.nii.zip"]
    zip_hits = [p for p in zip_candidates if p.exists()]
    zip_note = ""
    if zip_hits:
        zip_note = "\nFound .zip files. Extract them before loading NIfTI data:\n" + "\n".join(map(str, zip_hits))

    raise FileNotFoundError(
        f"Could not find modality '{modality}' for {patient_id} in {patient_dir}. Tried patterns:\n"
        + "\n".join(tried)
        + zip_note
    )


def load_volume(patient_dir, patient_id, modality):
    path = find_volume(patient_dir, patient_id, modality)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data, img

def resample_volume(volume, original_spacing, target_spacing=(1,1,1), order=1):
    zoom_factors = [o/t for o,t in zip(original_spacing, target_spacing)]
    return nd_zoom(volume, zoom_factors, order=order)