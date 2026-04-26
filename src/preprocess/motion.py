from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_frame(frame_dir: Path, frame_index: int) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = frame_dir / f"{frame_index}{extension}"
        if candidate.exists():
            return candidate
    if frame_dir.exists():
        for candidate in frame_dir.iterdir():
            if candidate.is_file() and candidate.stem == str(frame_index):
                return candidate
    return None


def read_rgb(path: Path, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def read_depth(path: Path, image_size: int) -> np.ndarray | None:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth = cv2.resize(depth, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32)


class OpticalFlowEngine:
    def __init__(self, algorithm: str = "TV-L1") -> None:
        self.algorithm = algorithm.upper()
        if self.algorithm == "TV-L1":
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        elif self.algorithm == "FARNEBACK":
            self.tvl1 = None
        else:
            raise ValueError(f"Unsupported flow algorithm: {algorithm}")

    def compute(self, onset_bgr: np.ndarray, apex_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        onset_gray = cv2.cvtColor(onset_bgr, cv2.COLOR_BGR2GRAY)
        apex_gray = cv2.cvtColor(apex_bgr, cv2.COLOR_BGR2GRAY)
        if self.algorithm == "TV-L1":
            flow = self.tvl1.calc(onset_gray, apex_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                onset_gray,
                apex_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        return flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32)


def build_flow_tensor(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.square(u) + np.square(v))
    scale = float(max(np.percentile(magnitude, 95), 1e-3))
    u_norm = np.clip(u / scale, -3.0, 3.0) / 3.0
    v_norm = np.clip(v / scale, -3.0, 3.0) / 3.0
    mag_norm = np.clip(magnitude / scale, 0.0, 3.0) / 3.0
    orientation = np.arctan2(v, u) / np.pi
    tensor = np.stack([u_norm, v_norm, mag_norm, orientation], axis=0)
    return tensor.astype(np.float32)


def build_uv_tensor(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.square(u) + np.square(v))
    scale = float(max(np.percentile(magnitude, 95), 1e-3))
    u_norm = np.clip(u / scale, -3.0, 3.0) / 3.0
    v_norm = np.clip(v / scale, -3.0, 3.0) / 3.0
    return np.stack([u_norm, v_norm], axis=0).astype(np.float32)


def build_depth_delta(onset_depth: np.ndarray, apex_depth: np.ndarray) -> np.ndarray:
    valid = (onset_depth > 0) & (apex_depth > 0)
    delta = np.zeros_like(onset_depth, dtype=np.float32)
    delta[valid] = apex_depth[valid] - onset_depth[valid]
    if np.any(valid):
        scale = float(max(np.percentile(np.abs(delta[valid]), 95), 1.0))
    else:
        scale = 1.0
    delta = np.clip(delta / scale, -3.0, 3.0) / 3.0
    return delta.astype(np.float32)


def build_uvd_tensor(u: np.ndarray, v: np.ndarray, onset_depth: np.ndarray, apex_depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uv = build_uv_tensor(u, v)
    depth = build_depth_delta(onset_depth, apex_depth)[None, :, :]
    uvd = np.concatenate([uv, depth], axis=0)
    return uv.astype(np.float32), depth.astype(np.float32), uvd.astype(np.float32)
