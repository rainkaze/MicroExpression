from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.motion import (
    OpticalFlowEngine,
    build_uvd_tensor,
    find_frame,
    read_depth,
    read_rgb,
)
from src.utils.runtime import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean CAS(ME)^3 scene-flow tensors for recognition.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "casme3_scene_flow" / "casme3_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "casme3_scene_flow",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--algorithm", choices=["TV-L1", "Farneback"], default="TV-L1")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    issue_column = "recognition_issues" if "recognition_issues" in df.columns else "issues"
    df = df[df[issue_column].fillna("").astype(str) == ""].copy()

    uv_dir = ensure_dir(args.output_dir / "uv")
    depth_dir_out = ensure_dir(args.output_dir / "depth")
    uvd_dir = ensure_dir(args.output_dir / "uvd")
    engine = OpticalFlowEngine(algorithm=args.algorithm)

    processed_rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for row in tqdm(df.to_dict("records"), desc="CAS(ME)^3 scene flow"):
        sample_id = str(row["sample_id"])
        frame_dir = PROJECT_ROOT / str(row["frame_dir"])
        depth_dir = PROJECT_ROOT / str(row["depth_dir"])
        onset = int(float(row["onset"]))
        apex = int(float(row["apex"]))

        onset_frame = find_frame(frame_dir, onset)
        apex_frame = find_frame(frame_dir, apex)
        onset_depth_path = find_frame(depth_dir, onset)
        apex_depth_path = find_frame(depth_dir, apex)
        if onset_frame is None or apex_frame is None:
            failures.append((sample_id, "missing_rgb_frame"))
            continue
        if onset_depth_path is None or apex_depth_path is None:
            failures.append((sample_id, "missing_depth_frame"))
            continue

        onset_rgb = read_rgb(onset_frame, args.image_size)
        apex_rgb = read_rgb(apex_frame, args.image_size)
        onset_depth = read_depth(onset_depth_path, args.image_size)
        apex_depth = read_depth(apex_depth_path, args.image_size)
        if onset_rgb is None or apex_rgb is None:
            failures.append((sample_id, "rgb_read_failed"))
            continue
        if onset_depth is None or apex_depth is None:
            failures.append((sample_id, "depth_read_failed"))
            continue

        u, v = engine.compute(onset_rgb, apex_rgb)
        uv_tensor, depth_tensor, uvd_tensor = build_uvd_tensor(u, v, onset_depth, apex_depth)

        uv_path = uv_dir / f"{sample_id}.npy"
        depth_path = depth_dir_out / f"{sample_id}.npy"
        uvd_path = uvd_dir / f"{sample_id}.npy"
        np.save(uv_path, uv_tensor.astype(np.float16))
        np.save(depth_path, depth_tensor.astype(np.float16))
        np.save(uvd_path, uvd_tensor.astype(np.float16))

        row["uv_path"] = str(uv_path.relative_to(PROJECT_ROOT))
        row["depth_path"] = str(depth_path.relative_to(PROJECT_ROOT))
        row["uvd_path"] = str(uvd_path.relative_to(PROJECT_ROOT))
        processed_rows.append(row)

    output_dir = ensure_dir(args.output_dir)
    manifest_out = output_dir / "casme3_scene_flow_manifest.csv"
    pd.DataFrame(processed_rows).to_csv(manifest_out, index=False, encoding="utf-8-sig")

    report_lines = [
        "# CAS(ME)^3 Scene-Flow Build Report",
        "",
        f"- Input recognition-clean rows: {len(df)}",
        f"- Processed rows: {len(processed_rows)}",
        f"- Failures: {len(failures)}",
        f"- Image size: {args.image_size}",
        f"- Optical-flow algorithm: {args.algorithm}",
        "",
        "## Outputs",
        "",
        "- `uv/`: 2-channel tensor `(u, v)`",
        "- `depth/`: 1-channel tensor `(depth_delta)`",
        "- `uvd/`: 3-channel tensor `(u, v, depth_delta)`",
        "- `casme3_scene_flow_manifest.csv`: training manifest with tensor paths",
        "",
        "## First Failures",
    ]
    for sample_id, reason in failures[:20]:
        report_lines.append(f"- {sample_id}: {reason}")
    (output_dir / "build_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved processed manifest: {manifest_out}")
    print(f"Processed rows: {len(processed_rows)}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
