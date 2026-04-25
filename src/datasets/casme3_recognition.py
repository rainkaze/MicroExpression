from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

LABEL_MODES: dict[str, list[str]] = {
    "7class": ["disgust", "surprise", "others", "fear", "anger", "sad", "happy"],
    "4class": ["negative", "positive", "surprise", "others"],
    "negative4": ["disgust", "fear", "anger", "sad"],
}


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _resolve_existing_path(project_root: Path, relative_path: str) -> Path:
    rel = _normalize_path(relative_path)
    stripped_data = rel.removeprefix("data/")
    candidates = [
        project_root / rel,
        project_root / "data" / rel,
        project_root / "data" / "processed" / rel.removeprefix("processed/"),
        project_root / "data" / "raw" / stripped_data,
        project_root / "data" / "interim" / rel.removeprefix("interim/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve path from manifest: {relative_path}")


def _find_frame(frame_dir: Path, frame_index: int) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = frame_dir / f"{frame_index}{extension}"
        if candidate.exists():
            return candidate
    for candidate in frame_dir.iterdir():
        if candidate.is_file() and candidate.stem == str(frame_index):
            return candidate
    raise FileNotFoundError(f"Could not find frame {frame_index} in {frame_dir}")


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    subject: str
    label_index: int
    label_name: str
    input_path: Path
    raw_row: dict[str, Any]


class CASME3RecognitionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        input_mode: str = "flow",
        label_mode: str = "7class",
        augment: bool = False,
        clean_only: bool = True,
        indices: list[int] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.input_mode = input_mode
        self.label_mode = label_mode
        self.augment = augment

        if label_mode not in LABEL_MODES:
            raise ValueError(f"Unsupported label_mode: {label_mode}")
        if input_mode not in {"flow", "rgbd_flow", "apex_rgb"}:
            raise ValueError(f"Unsupported input_mode: {input_mode}")

        df = pd.read_csv(self.manifest_path)
        if clean_only and "issues" in df.columns:
            df = df[df["issues"].fillna("").astype(str) == ""].copy()
        if indices is not None:
            df = df.iloc[indices].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        labels = LABEL_MODES[label_mode]
        mapping = {name: index for index, name in enumerate(labels)}
        label_column = "emotion_4" if label_mode == "4class" else "emotion_7"

        self.records: list[SampleRecord] = []
        for row in df.to_dict("records"):
            label_name = str(row[label_column]).strip().lower()
            if label_name not in mapping:
                continue
            input_path = self._resolve_input_path(row)
            self.records.append(
                SampleRecord(
                    sample_id=str(row["sample_id"]),
                    subject=str(row["subject"]),
                    label_index=mapping[label_name],
                    label_name=label_name,
                    input_path=input_path,
                    raw_row=row,
                )
            )

        if not self.records:
            raise RuntimeError("No records were loaded for CAS(ME)^3 recognition.")

    def _resolve_input_path(self, row: dict[str, Any]) -> Path:
        if self.input_mode == "flow":
            return _resolve_existing_path(self.project_root, str(row["flow_path"]))
        if self.input_mode == "rgbd_flow":
            return _resolve_existing_path(self.project_root, str(row["rgbd_flow_path"]))
        frame_dir = _resolve_existing_path(self.project_root, str(row["frame_dir"]))
        apex = int(float(row["apex"]))
        return _find_frame(frame_dir, apex)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _augment_motion(tensor: np.ndarray) -> np.ndarray:
        tensor = tensor.copy()
        if np.random.rand() < 0.5:
            tensor = np.flip(tensor, axis=2).copy()
            if tensor.shape[0] >= 1:
                tensor[0] = -tensor[0]
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.92, 1.08)
            tensor[: min(3, tensor.shape[0])] *= scale
        if np.random.rand() < 0.15 and tensor.shape[0] >= 2:
            noise = np.random.normal(0.0, 0.015, size=tensor[:2].shape).astype(np.float32)
            tensor[:2] += noise
        return np.clip(tensor, -1.0, 1.0)

    @staticmethod
    def _load_rgb(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.input_mode == "apex_rgb":
            data = self._load_rgb(record.input_path)
        else:
            array = np.load(record.input_path).astype(np.float32)
            if self.augment:
                array = self._augment_motion(array)
            data = torch.from_numpy(np.ascontiguousarray(array))
        return {
            "input": data,
            "label": record.label_index,
            "label_name": record.label_name,
            "subject": record.subject,
            "sample_id": record.sample_id,
        }
