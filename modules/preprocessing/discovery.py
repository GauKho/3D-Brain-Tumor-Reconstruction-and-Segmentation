"""Discover valid BraTS cases on disk."""

from pathlib import Path
from typing import Dict, List, Optional


def find_modality_file(case_dir: Path, modality: str) -> Optional[Path]:
    matches = []
    for pattern in (f"*_{modality}.nii", f"*_{modality}.nii.gz", f"*{modality}.nii", f"*{modality}.nii.gz"):
        matches.extend(case_dir.glob(pattern))
    matches = sorted(set(matches))
    return matches[0] if matches else None


def discover_brats_cases(root: Path, modalities=("t1ce", "t2", "flair")) -> List[Dict[str, str]]:
    records = []
    seen = set()
    for seg_path in sorted(Path(root).rglob("*_seg.nii*")):
        case_dir = seg_path.parent
        if case_dir in seen:
            continue
        seen.add(case_dir)
        record = {"case_id": case_dir.name, "case_dir": str(case_dir), "seg": str(seg_path)}
        files = {modality: find_modality_file(case_dir, modality) for modality in modalities}
        if all(files.values()):
            record.update({name: str(path) for name, path in files.items()})
            records.append(record)
    return records
