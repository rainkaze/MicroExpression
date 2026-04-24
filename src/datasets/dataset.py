import os
import random
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EMOTION_CLASSES = [
    "happiness",
    "disgust",
    "surprise",
    "repression",
    "sadness",
    "fear",
    "others",
]

EMOTION_TO_LABEL = {name: idx for idx, name in enumerate(EMOTION_CLASSES)}
EMOTION_ALIASES = {
    "repressed happiness": "happiness",
    "tense": "others",
}
COARSE_CLASSES = ["positive", "negative", "surprise", "others"]
COARSE_TO_LABEL = {name: idx for idx, name in enumerate(COARSE_CLASSES)}
FINE_TO_COARSE = {
    "happiness": "positive",
    "disgust": "negative",
    "repression": "negative",
    "sadness": "negative",
    "fear": "negative",
    "surprise": "surprise",
    "others": "others",
}


def normalize_emotion_name(emotion: str) -> str:
    normalized = str(emotion).strip().lower()
    return EMOTION_ALIASES.get(normalized, normalized)


class CASME2FlowDataset(Dataset):
    def __init__(
        self,
        processed_dir: str,
        csv_path: str,
        subjects: Iterable[str] | None = None,
        augment: bool = False,
        target_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.processed_dir = processed_dir
        self.augment = augment
        self.target_size = target_size

        df = pd.read_excel(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        df = df[pd.to_numeric(df["OnsetFrame"], errors="coerce").notnull()].copy()
        df["Subject"] = df["Subject"].apply(lambda x: str(int(x)).zfill(2))
        df["Filename"] = df["Filename"].astype(str).str.strip()
        df["Estimated Emotion"] = df["Estimated Emotion"].apply(normalize_emotion_name)

        if subjects is not None:
            subjects = set(subjects)
            df = df[df["Subject"].isin(subjects)].copy()

        valid_rows = []
        missing_samples = []
        unknown_labels = []

        for row in df.to_dict("records"):
            emotion = row["Estimated Emotion"]
            if emotion not in EMOTION_TO_LABEL:
                unknown_labels.append(emotion)
                continue

            sample_id = f"sub{row['Subject']}_{row['Filename']}"
            u_path = os.path.join(self.processed_dir, "u", f"{sample_id}.npy")
            v_path = os.path.join(self.processed_dir, "v", f"{sample_id}.npy")
            if not os.path.exists(u_path) or not os.path.exists(v_path):
                missing_samples.append(sample_id)
                continue

            row["u_path"] = u_path
            row["v_path"] = v_path
            row["label"] = EMOTION_TO_LABEL[emotion]
            row["coarse_label"] = COARSE_TO_LABEL[FINE_TO_COARSE[emotion]]
            valid_rows.append(row)

        if not valid_rows:
            raise RuntimeError("No valid samples were found for the requested split.")

        self.samples = valid_rows
        self.missing_samples = missing_samples
        self.unknown_labels = sorted(set(unknown_labels))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_flow(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            u = np.flip(u, axis=1).copy()
            v = np.flip(v, axis=1).copy()
            u = -u

        if random.random() < 0.3:
            scale = random.uniform(0.9, 1.1)
            u = u * scale
            v = v * scale

        if random.random() < 0.2:
            noise_std = random.uniform(0.0, 0.01)
            u = u + np.random.normal(0.0, noise_std, size=u.shape).astype(np.float32)
            v = v + np.random.normal(0.0, noise_std, size=v.shape).astype(np.float32)

        return u, v

    @staticmethod
    def _build_representation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        magnitude = np.sqrt(np.square(u) + np.square(v))
        scale = np.percentile(magnitude, 95)
        scale = float(max(scale, 1e-3))

        u = np.clip(u / scale, -3.0, 3.0) / 3.0
        v = np.clip(v / scale, -3.0, 3.0) / 3.0
        magnitude = np.clip(magnitude / scale, 0.0, 3.0) / 3.0

        orientation = np.arctan2(v, u) / np.pi
        representation = np.stack([u, v, magnitude, orientation], axis=0).astype(np.float32)
        return representation

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.samples[index]
        u = np.load(row["u_path"]).astype(np.float32)
        v = np.load(row["v_path"]).astype(np.float32)

        if u.shape != self.target_size or v.shape != self.target_size:
            raise ValueError(
                f"Unexpected flow shape for {row['u_path']}: {u.shape}, {v.shape}"
            )

        if self.augment:
            u, v = self._augment_flow(u, v)

        flow = self._build_representation(u, v)
        flow_tensor = torch.from_numpy(np.ascontiguousarray(flow))
        return flow_tensor, int(row["label"]), int(row["coarse_label"])
