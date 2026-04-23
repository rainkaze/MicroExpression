from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess.flow_utils import OpticalFlowEngine


PROJECT_ROOT = Path(__file__).resolve().parent


def read_rgb(path: Path, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def find_frame(frame_dir: Path, frame_index: int) -> Path | None:
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = frame_dir / f"{frame_index}{extension}"
        if candidate.exists():
            return candidate
    target = str(frame_index)
    for candidate in frame_dir.iterdir() if frame_dir.exists() else []:
        if candidate.is_file() and candidate.stem == target:
            return candidate
    return None


def build_flow_tensor(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.square(u) + np.square(v))
    scale = float(max(np.percentile(magnitude, 95), 1e-3))
    u_norm = np.clip(u / scale, -3.0, 3.0) / 3.0
    v_norm = np.clip(v / scale, -3.0, 3.0) / 3.0
    mag_norm = np.clip(magnitude / scale, 0.0, 3.0) / 3.0
    orientation = np.arctan2(v, u) / np.pi
    return np.stack([u_norm, v_norm, mag_norm, orientation], axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CAS(ME)^3 Part A ME clips into onset-to-apex flow tensors.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "reports" / "casme3_part_a_me_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "processed" / "casme3_flow")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--algorithm", choices=["TV-L1", "Farneback"], default="TV-L1")
    parser.add_argument("--include-issues", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if not args.include_issues:
        df = df[df["issues"].fillna("").astype(str) == ""].copy()

    flow_dir = args.output_dir / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)
    engine = OpticalFlowEngine(algorithm=args.algorithm)

    rows = []
    failures = []
    for row in tqdm(df.to_dict("records"), desc="CAS(ME)^3 flow"):
        sample_id = str(row["sample_id"])
        frame_dir = PROJECT_ROOT / str(row["frame_dir"])
        onset = int(float(row["onset"]))
        apex = int(float(row["apex"]))
        onset_path = find_frame(frame_dir, onset)
        apex_path = find_frame(frame_dir, apex)
        if onset_path is None or apex_path is None:
            failures.append((sample_id, "missing_onset_or_apex"))
            continue

        onset_image = read_rgb(onset_path, args.image_size)
        apex_image = read_rgb(apex_path, args.image_size)
        if onset_image is None or apex_image is None:
            failures.append((sample_id, "image_read_failed"))
            continue

        u, v = engine.compute_flow(onset_image, apex_image)
        flow = build_flow_tensor(u.astype(np.float32), v.astype(np.float32))
        flow_path = flow_dir / f"{sample_id}.npy"
        np.save(flow_path, flow.astype(np.float16))
        row["flow_path"] = str(flow_path.relative_to(PROJECT_ROOT))
        rows.append(row)

    output_manifest = args.output_dir / "casme3_flow_manifest.csv"
    pd.DataFrame(rows).to_csv(output_manifest, index=False, encoding="utf-8-sig")
    print(f"Saved flow tensors: {len(rows)}")
    print(f"Failed: {len(failures)}")
    if failures:
        print(f"First failures: {failures[:10]}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
