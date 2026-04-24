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
    build_depth_delta,
    build_flow_tensor,
    find_frame,
    read_depth,
    read_rgb,
)
from src.utils.runtime import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fresh motion tensors for CAS(ME)^3 recognition.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "casme3_recognition_v2" / "casme3_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "casme3_recognition_v2",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--algorithm", choices=["TV-L1", "Farneback"], default="TV-L1")
    parser.add_argument("--include-issues", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if not args.include_issues:
        df = df[df["issues"].fillna("").astype(str) == ""].copy()

    flow_dir = ensure_dir(args.output_dir / "flow")
    rgbd_dir = ensure_dir(args.output_dir / "rgbd_flow")
    engine = OpticalFlowEngine(algorithm=args.algorithm)

    processed_rows = []
    failures: list[tuple[str, str]] = []

    for row in tqdm(df.to_dict("records"), desc="CAS(ME)^3 motion"):
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

        onset_rgb = read_rgb(onset_frame, args.image_size)
        apex_rgb = read_rgb(apex_frame, args.image_size)
        if onset_rgb is None or apex_rgb is None:
            failures.append((sample_id, "rgb_read_failed"))
            continue

        u, v = engine.compute(onset_rgb, apex_rgb)
        flow_tensor = build_flow_tensor(u, v)
        flow_path = flow_dir / f"{sample_id}.npy"
        np.save(flow_path, flow_tensor.astype(np.float16))

        row["flow_path"] = str(flow_path.relative_to(PROJECT_ROOT))

        if onset_depth_path is not None and apex_depth_path is not None:
            onset_depth = read_depth(onset_depth_path, args.image_size)
            apex_depth = read_depth(apex_depth_path, args.image_size)
            if onset_depth is not None and apex_depth is not None:
                depth_delta = build_depth_delta(onset_depth, apex_depth)
                rgbd_tensor = np.concatenate([flow_tensor, depth_delta[None, :, :]], axis=0)
                rgbd_path = rgbd_dir / f"{sample_id}.npy"
                np.save(rgbd_path, rgbd_tensor.astype(np.float16))
                row["rgbd_flow_path"] = str(rgbd_path.relative_to(PROJECT_ROOT))
            else:
                failures.append((sample_id, "depth_read_failed"))
                row["rgbd_flow_path"] = ""
        else:
            row["rgbd_flow_path"] = ""

        processed_rows.append(row)

    processed_manifest = pd.DataFrame(processed_rows)
    manifest_out = args.output_dir / "casme3_recognition_manifest.csv"
    processed_manifest.to_csv(manifest_out, index=False, encoding="utf-8-sig")

    audit_lines = [
        "# CAS(ME)^3 Motion Build Report",
        "",
        f"- Input rows: {len(df)}",
        f"- Processed rows: {len(processed_rows)}",
        f"- Failures: {len(failures)}",
        "",
        "## First Failures",
    ]
    for sample_id, reason in failures[:20]:
        audit_lines.append(f"- {sample_id}: {reason}")
    (args.output_dir / "build_report.md").write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"Saved processed manifest: {manifest_out}")
    print(f"Processed rows: {len(processed_rows)}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
