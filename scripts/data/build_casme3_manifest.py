from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.motion import find_frame
from src.utils.runtime import ensure_dir


EMOTION_4_MAP = {
    "happy": "positive",
    "disgust": "negative",
    "fear": "negative",
    "anger": "negative",
    "sad": "negative",
    "surprise": "surprise",
    "others": "others",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean CAS(ME)^3 Part A ME manifest for recognition.")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "CAS(ME)^3" / "annotation" / "cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx",
    )
    parser.add_argument(
        "--clip-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "CAS(ME)^3" / "Part_A_ME_clip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "casme3_recognition_v2",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.annotation)
    output_dir = ensure_dir(args.output_dir)

    rows = []
    for row in df.to_dict("records"):
        subject = str(row["Subject"]).strip()
        video_code = str(row["Filename"]).strip()
        onset = int(row["Onset"])
        apex = int(row["Apex"])
        offset = int(row["Offset"])
        emotion_7 = str(row["emotion"]).strip().lower()
        emotion_4 = EMOTION_4_MAP.get(emotion_7, "")
        objective_class = str(row["Objective class"]).strip()
        au = "" if pd.isna(row["AU"]) else str(row["AU"]).strip()
        sample_id = f"{subject}_{video_code}_{onset}"

        frame_dir = args.clip_root / "frame" / sample_id
        depth_dir = args.clip_root / "depth" / sample_id
        video_path = args.clip_root / "video" / f"{sample_id}.mp4"

        issues: list[str] = []
        if not frame_dir.exists():
            issues.append("missing_frame_dir")
        if not depth_dir.exists():
            issues.append("missing_depth_dir")
        if not video_path.exists():
            issues.append("missing_video")
        if offset < onset:
            issues.append("offset_before_onset")
        if not (onset <= apex <= offset):
            issues.append("apex_outside_onset_offset")

        onset_frame = find_frame(frame_dir, onset) if frame_dir.exists() else None
        apex_frame = find_frame(frame_dir, apex) if frame_dir.exists() else None
        offset_frame = find_frame(frame_dir, offset) if frame_dir.exists() else None
        onset_depth = find_frame(depth_dir, onset) if depth_dir.exists() else None
        apex_depth = find_frame(depth_dir, apex) if depth_dir.exists() else None
        offset_depth = find_frame(depth_dir, offset) if depth_dir.exists() else None

        if onset_frame is None:
            issues.append("missing_onset_frame")
        if apex_frame is None:
            issues.append("missing_apex_frame")
        if offset_frame is None:
            issues.append("missing_offset_frame")
        if onset_depth is None or apex_depth is None or offset_depth is None:
            issues.append("missing_depth_frame")

        frame_files = sorted([x for x in frame_dir.iterdir() if x.is_file()]) if frame_dir.exists() else []
        depth_files = sorted([x for x in depth_dir.iterdir() if x.is_file()]) if depth_dir.exists() else []
        frame_depth_name_match = [x.stem for x in frame_files] == [x.stem for x in depth_files] if frame_files and depth_files else False
        if frame_files and depth_files and not frame_depth_name_match:
            issues.append("frame_depth_names_mismatch")
        if len(frame_files) <= 2:
            issues.append("very_short_clip")

        rows.append(
            {
                "sample_id": sample_id,
                "subject": subject,
                "video_code": video_code,
                "onset": onset,
                "apex": apex,
                "offset": offset,
                "emotion_7": emotion_7,
                "emotion_4": emotion_4,
                "objective_class": objective_class,
                "au": au,
                "frame_dir": str(frame_dir.relative_to(PROJECT_ROOT)),
                "depth_dir": str(depth_dir.relative_to(PROJECT_ROOT)),
                "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                "frame_count": len(frame_files),
                "depth_count": len(depth_files),
                "frame_depth_name_match": frame_depth_name_match,
                "onset_frame_exists": onset_frame is not None,
                "apex_frame_exists": apex_frame is not None,
                "offset_frame_exists": offset_frame is not None,
                "issues": ";".join(sorted(set(issues))),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "casme3_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    clean = manifest[manifest["issues"].fillna("").astype(str) == ""].copy()
    audit_path = output_dir / "casme3_manifest_audit.md"
    audit_lines = [
        "# CAS(ME)^3 Recognition Manifest Audit",
        "",
        f"- Samples: {len(manifest)}",
        f"- Clean samples: {len(clean)}",
        f"- Subjects: {manifest['subject'].nunique()}",
        "",
        "## 7-Class Distribution",
    ]
    for label, count in clean["emotion_7"].value_counts().sort_index().items():
        audit_lines.append(f"- {label}: {count}")
    audit_lines.extend(["", "## 4-Class Distribution"])
    for label, count in clean["emotion_4"].value_counts().sort_index().items():
        audit_lines.append(f"- {label}: {count}")
    audit_lines.extend(["", "## Issue Distribution"])
    issue_counts: dict[str, int] = {}
    for issue_text in manifest["issues"].fillna("").astype(str):
        for issue in [item for item in issue_text.split(";") if item]:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
    for issue, count in sorted(issue_counts.items(), key=lambda item: (-item[1], item[0])):
        audit_lines.append(f"- {issue}: {count}")
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved audit: {audit_path}")
    print(f"Clean samples: {len(clean)} / {len(manifest)}")


if __name__ == "__main__":
    main()
