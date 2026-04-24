from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMOTION_7 = ["disgust", "surprise", "others", "fear", "anger", "sad", "happy"]
EMOTION_4_MAP = {
    "happy": "positive",
    "disgust": "negative",
    "fear": "negative",
    "anger": "negative",
    "sad": "negative",
    "surprise": "surprise",
    "others": "others",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SampleAudit:
    sample_id: str
    subject: str
    video_code: str
    onset: int
    apex: int
    offset: int
    emotion_7: str
    emotion_4: str
    objective_class: str
    au: str
    frame_dir: str
    depth_dir: str
    video_path: str
    frame_count: int
    depth_count: int
    frame_depth_name_match: bool
    onset_frame_exists: bool
    apex_frame_exists: bool
    offset_frame_exists: bool
    first_frame_index: int | None
    last_frame_index: int | None
    rgb_size: str
    depth_size: str
    rgb_mode: str
    depth_mode: str
    issues: str


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_emotion(value: object) -> str:
    emotion = normalize_text(value).lower()
    aliases = {
        "happiness": "happy",
        "surprised": "surprise",
        "other": "others",
    }
    return aliases.get(emotion, emotion)


def parse_int(value: object) -> int:
    if pd.isna(value):
        raise ValueError("missing integer value")
    return int(float(value))


def numeric_stem(path: Path) -> int | None:
    match = re.search(r"\d+", path.stem)
    if match is None:
        return None
    return int(match.group(0))


def list_images(folder: Path | None) -> list[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: (numeric_stem(p) is None, numeric_stem(p) or 0, p.name.lower()),
    )


def build_case_insensitive_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {p.name.lower(): p for p in root.iterdir() if p.is_dir()}


def find_dir(index: dict[str, Path], sample_id: str) -> Path | None:
    return index.get(sample_id.lower())


def find_video(video_root: Path, sample_id: str) -> Path | None:
    if not video_root.exists():
        return None
    exact = video_root / f"{sample_id}.mp4"
    if exact.exists():
        return exact
    target = f"{sample_id}.mp4".lower()
    for path in video_root.iterdir():
        if path.is_file() and path.name.lower() == target:
            return path
    return None


def probe_image(path: Path | None) -> tuple[str, str]:
    if path is None or not path.exists():
        return "", ""
    try:
        with Image.open(path) as image:
            return f"{image.size[0]}x{image.size[1]}", image.mode
    except Exception as exc:
        return "", f"open_error:{exc}"


def frame_exists(images: list[Path], frame_index: int) -> bool:
    return any(numeric_stem(path) == frame_index for path in images)


def read_annotation(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(col).strip() for col in df.columns]
    required = ["Subject", "Filename", "Onset", "Apex", "Offset", "AU", "Objective class", "emotion"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing annotation columns: {missing}")
    return df


def audit_samples(data_root: Path, annotation_path: Path) -> list[SampleAudit]:
    clip_root = data_root / "Part_A_ME_clip"
    frame_root = clip_root / "frame"
    depth_root = clip_root / "depth"
    video_root = clip_root / "video"
    frame_index = build_case_insensitive_index(frame_root)
    depth_index = build_case_insensitive_index(depth_root)
    annotation = read_annotation(annotation_path)

    samples: list[SampleAudit] = []
    for _, row in annotation.iterrows():
        subject = normalize_text(row["Subject"])
        video_code = normalize_text(row["Filename"])
        onset = parse_int(row["Onset"])
        apex = parse_int(row["Apex"])
        offset = parse_int(row["Offset"])
        emotion_7 = normalize_emotion(row["emotion"])
        emotion_4 = EMOTION_4_MAP.get(emotion_7, "unknown")
        sample_id = f"{subject}_{video_code}_{onset}"

        issues: list[str] = []
        if emotion_7 not in EMOTION_7:
            issues.append(f"unknown_emotion:{emotion_7}")
        if offset < onset:
            issues.append("offset_before_onset")
        if not (onset <= apex <= offset):
            issues.append("apex_outside_onset_offset")

        frame_dir = find_dir(frame_index, sample_id)
        depth_dir = find_dir(depth_index, sample_id)
        video_path = find_video(video_root, sample_id)
        if frame_dir is None:
            issues.append("missing_frame_dir")
        if depth_dir is None:
            issues.append("missing_depth_dir")
        if video_path is None:
            issues.append("missing_video")

        frame_images = list_images(frame_dir)
        depth_images = list_images(depth_dir)
        frame_names = {path.stem.lower() for path in frame_images}
        depth_names = {path.stem.lower() for path in depth_images}
        name_match = frame_names == depth_names
        if not name_match:
            issues.append("frame_depth_names_mismatch")
        if len(frame_images) < 3:
            issues.append("very_short_clip")

        onset_exists = frame_exists(frame_images, onset)
        apex_exists = frame_exists(frame_images, apex)
        offset_exists = frame_exists(frame_images, offset)
        if not onset_exists:
            issues.append("missing_onset_frame")
        if not apex_exists:
            issues.append("missing_apex_frame")
        if not offset_exists:
            issues.append("missing_offset_frame")

        frame_numbers = [numeric_stem(path) for path in frame_images]
        frame_numbers = [num for num in frame_numbers if num is not None]
        rgb_size, rgb_mode = probe_image(frame_images[0] if frame_images else None)
        depth_size, depth_mode = probe_image(depth_images[0] if depth_images else None)
        if rgb_size and depth_size and rgb_size != depth_size:
            issues.append("rgb_depth_size_mismatch")

        samples.append(
            SampleAudit(
                sample_id=sample_id,
                subject=subject,
                video_code=video_code,
                onset=onset,
                apex=apex,
                offset=offset,
                emotion_7=emotion_7,
                emotion_4=emotion_4,
                objective_class=normalize_text(row["Objective class"]),
                au=normalize_text(row["AU"]),
                frame_dir=str(frame_dir.relative_to(PROJECT_ROOT)) if frame_dir else "",
                depth_dir=str(depth_dir.relative_to(PROJECT_ROOT)) if depth_dir else "",
                video_path=str(video_path.relative_to(PROJECT_ROOT)) if video_path else "",
                frame_count=len(frame_images),
                depth_count=len(depth_images),
                frame_depth_name_match=name_match,
                onset_frame_exists=onset_exists,
                apex_frame_exists=apex_exists,
                offset_frame_exists=offset_exists,
                first_frame_index=min(frame_numbers) if frame_numbers else None,
                last_frame_index=max(frame_numbers) if frame_numbers else None,
                rgb_size=rgb_size,
                depth_size=depth_size,
                rgb_mode=rgb_mode,
                depth_mode=depth_mode,
                issues=";".join(issues),
            )
        )
    return samples


def describe_counts(counter: Counter[str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in counter.most_common())


def write_report(samples: list[SampleAudit], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(sample) for sample in samples]
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "casme3_part_a_me_manifest.csv", index=False, encoding="utf-8-sig")
    (output_dir / "casme3_part_a_me_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    frame_counts = df["frame_count"].tolist()
    issue_rows = df[df["issues"].astype(str) != ""]
    issue_counter: Counter[str] = Counter()
    for value in issue_rows["issues"]:
        for issue in str(value).split(";"):
            if issue:
                issue_counter[issue] += 1

    lines = [
        "# CAS(ME)^3 Part A ME Clip Audit",
        "",
        "## Summary",
        f"- Samples: {len(df)}",
        f"- Subjects: {df['subject'].nunique()}",
        f"- Videos codes: {df['video_code'].nunique()}",
        f"- Samples with any issue: {len(issue_rows)}",
        f"- Frame count min/median/mean/max: {min(frame_counts)} / {median(frame_counts):.1f} / {mean(frame_counts):.2f} / {max(frame_counts)}",
        f"- RGB sizes: {dict(Counter(df['rgb_size']))}",
        f"- Depth sizes: {dict(Counter(df['depth_size']))}",
        f"- RGB modes: {dict(Counter(df['rgb_mode']))}",
        f"- Depth modes: {dict(Counter(df['depth_mode']))}",
        "",
        "## 7-Class Emotion Distribution",
        describe_counts(Counter(df["emotion_7"])),
        "",
        "## 4-Class Emotion Distribution",
        describe_counts(Counter(df["emotion_4"])),
        "",
        "## Objective Class Distribution",
        describe_counts(Counter(df["objective_class"])),
        "",
        "## Subject Sample Distribution",
        f"- Min samples per subject: {min(Counter(df['subject']).values())}",
        f"- Max samples per subject: {max(Counter(df['subject']).values())}",
        f"- Subjects with one sample: {sum(1 for count in Counter(df['subject']).values() if count == 1)}",
        "",
        "## Issue Distribution",
        describe_counts(issue_counter) if issue_counter else "- No issues found.",
        "",
        "## First 20 Issue Rows",
    ]

    if len(issue_rows) == 0:
        lines.append("- No issue rows.")
    else:
        for _, row in issue_rows.head(20).iterrows():
            lines.append(f"- {row['sample_id']}: {row['issues']}")

    (output_dir / "casme3_part_a_me_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CAS(ME)^3 Part A ME clip files and labels.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "CAS(ME)^3")
    parser.add_argument(
        "--annotation",
        type=Path,
        default=PROJECT_ROOT / "data" / "CAS(ME)^3" / "annotation" / "cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    samples = audit_samples(args.data_root, args.annotation)
    write_report(samples, args.output_dir)
    print(f"Audited {len(samples)} samples")
    print(f"Report: {args.output_dir / 'casme3_part_a_me_audit.md'}")
    print(f"Manifest: {args.output_dir / 'casme3_part_a_me_manifest.csv'}")


if __name__ == "__main__":
    main()
