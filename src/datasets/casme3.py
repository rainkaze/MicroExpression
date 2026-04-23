from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


CASME3_EMOTIONS_7 = ["disgust", "surprise", "others", "fear", "anger", "sad", "happy"]
CASME3_EMOTION_TO_LABEL = {name: index for index, name in enumerate(CASME3_EMOTIONS_7)}


class CASME3ApexDataset(Dataset):
    """CAS(ME)^3 Part A micro-expression dataset using one RGB apex frame per sample."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        transform: Callable | None = None,
        clean_only: bool = True,
        subjects: set[str] | list[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.transform = transform

        df = pd.read_csv(self.manifest_path)
        if clean_only:
            df = df[df["issues"].fillna("").astype(str) == ""].copy()
        if subjects is not None:
            subject_set = set(subjects)
            df = df[df["subject"].isin(subject_set)].copy()

        samples = []
        for row in df.to_dict("records"):
            emotion = str(row["emotion_7"]).strip().lower()
            if emotion not in CASME3_EMOTION_TO_LABEL:
                continue
            apex = int(float(row["apex"]))
            frame_dir = self.project_root / str(row["frame_dir"])
            image_path = self._find_frame(frame_dir, apex)
            if image_path is None:
                continue
            row["image_path"] = str(image_path)
            row["label"] = CASME3_EMOTION_TO_LABEL[emotion]
            samples.append(row)

        if not samples:
            raise RuntimeError("No CAS(ME)^3 apex samples were found.")
        self.samples = samples

    @staticmethod
    def _find_frame(frame_dir: Path, frame_index: int) -> Path | None:
        for extension in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = frame_dir / f"{frame_index}{extension}"
            if candidate.exists():
                return candidate
        target = str(frame_index)
        for candidate in frame_dir.iterdir() if frame_dir.exists() else []:
            if candidate.is_file() and candidate.stem == target:
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        row = self.samples[index]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "subject": str(row["subject"]),
            "sample_id": str(row["sample_id"]),
            "emotion": str(row["emotion_7"]),
        }


def load_manifest(manifest_path: str | Path, clean_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if clean_only:
        df = df[df["issues"].fillna("").astype(str) == ""].copy()
    return df


class CASME3FlowDataset(Dataset):
    """CAS(ME)^3 Part A micro-expression dataset using onset-to-apex flow tensors."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        augment: bool = False,
        subjects: set[str] | list[str] | None = None,
        path_column: str = "flow_path",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.augment = augment
        self.path_column = path_column

        df = pd.read_csv(self.manifest_path)
        if subjects is not None:
            subject_set = set(subjects)
            df = df[df["subject"].isin(subject_set)].copy()

        samples = []
        for row in df.to_dict("records"):
            emotion = str(row["emotion_7"]).strip().lower()
            if emotion not in CASME3_EMOTION_TO_LABEL:
                continue
            if path_column not in row or pd.isna(row[path_column]):
                continue
            flow_path = self.project_root / str(row[path_column])
            if not flow_path.exists():
                continue
            row["label"] = CASME3_EMOTION_TO_LABEL[emotion]
            row["motion_path"] = str(flow_path)
            samples.append(row)

        if not samples:
            raise RuntimeError("No CAS(ME)^3 flow samples were found.")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _augment_flow(flow: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            flow = np.flip(flow, axis=2).copy()
            flow[0] = -flow[0]
            flow[3] = np.arctan2(flow[1], flow[0]) / np.pi
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.90, 1.10)
            flow[0:3] = flow[0:3] * scale
        if np.random.rand() < 0.2:
            noise = np.random.normal(0.0, 0.01, size=flow[0:2].shape).astype(np.float32)
            flow[0:2] = flow[0:2] + noise
            magnitude = np.sqrt(np.square(flow[0]) + np.square(flow[1]))
            flow[2] = np.clip(magnitude, 0.0, 1.0)
            flow[3] = np.arctan2(flow[1], flow[0]) / np.pi
        return flow

    def __getitem__(self, index: int):
        row = self.samples[index]
        flow = np.load(row["motion_path"]).astype(np.float32)
        if self.augment:
            flow = self._augment_flow(flow)
        return {
            "flow": torch.from_numpy(np.ascontiguousarray(flow)),
            "label": int(row["label"]),
            "subject": str(row["subject"]),
            "sample_id": str(row["sample_id"]),
            "emotion": str(row["emotion_7"]),
        }
