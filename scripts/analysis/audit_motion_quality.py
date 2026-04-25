from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(value: str) -> Path:
    return PROJECT_ROOT / value.replace("\\", "/").strip()


def _motion_stats(flow_path: Path) -> dict[str, float]:
    tensor = np.load(flow_path).astype(np.float32)
    magnitude = tensor[2]
    u = tensor[0]
    v = tensor[1]
    uv_abs = np.sqrt(np.square(u) + np.square(v))
    return {
        "motion_mag_mean": float(np.mean(magnitude)),
        "motion_mag_p75": float(np.percentile(magnitude, 75)),
        "motion_mag_p95": float(np.percentile(magnitude, 95)),
        "motion_mag_max": float(np.max(magnitude)),
        "motion_uv_abs_mean": float(np.mean(uv_abs)),
    }


def audit(manifest_path: Path, output_dir: Path, weak_mean_threshold: float) -> None:
    df = pd.read_csv(manifest_path)
    rows = []
    for row in df.to_dict("records"):
        stats = {
            "motion_mag_mean": np.nan,
            "motion_mag_p75": np.nan,
            "motion_mag_p95": np.nan,
            "motion_mag_max": np.nan,
            "motion_uv_abs_mean": np.nan,
        }
        issues = str(row.get("issues", "") if not pd.isna(row.get("issues", "")) else "")
        flow_path = str(row.get("flow_path", ""))
        if not issues and flow_path:
            stats = _motion_stats(_resolve_path(flow_path))
        row.update(stats)
        onset = float(row.get("onset", 0))
        apex = float(row.get("apex", 0))
        offset = float(row.get("offset", 0))
        row["duration_onset_apex"] = apex - onset
        row["duration_onset_offset"] = offset - onset
        row["zero_motion"] = bool(row["motion_mag_mean"] == 0.0)
        row["weak_motion"] = bool(row["motion_mag_mean"] <= weak_mean_threshold)
        row["apex_equals_onset"] = bool(apex == onset)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    enriched_path = output_dir / "casme3_recognition_manifest_motion_audit.csv"
    out_df.to_csv(enriched_path, index=False, encoding="utf-8-sig")

    clean_df = out_df[out_df["issues"].fillna("").astype(str) == ""].copy()
    zero_df = clean_df[clean_df["zero_motion"]]
    weak_df = clean_df[clean_df["weak_motion"]]
    nonzero_df = clean_df[~clean_df["zero_motion"]].copy()
    nonzero_manifest_path = output_dir / "casme3_recognition_manifest_nonzero_motion.csv"
    nonzero_df.to_csv(nonzero_manifest_path, index=False, encoding="utf-8-sig")
    zero_df.to_csv(output_dir / "zero_motion_samples.csv", index=False, encoding="utf-8-sig")
    weak_df.to_csv(output_dir / "weak_motion_samples.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# CAS(ME)^3 Motion Quality Audit",
        "",
        f"- source manifest: `{manifest_path}`",
        f"- clean samples: `{len(clean_df)}`",
        f"- zero-motion clean samples: `{len(zero_df)}`",
        f"- weak-motion clean samples (mean <= {weak_mean_threshold}): `{len(weak_df)}`",
        f"- nonzero-motion clean samples: `{len(nonzero_df)}`",
        "",
        "## Zero-Motion Distribution",
        "",
    ]
    if len(zero_df) > 0:
        for label, count in zero_df["emotion_4"].value_counts().sort_index().items():
            lines.append(f"- 4-class `{label}`: `{count}`")
        lines.append("")
        for label, count in zero_df["emotion_7"].value_counts().sort_index().items():
            lines.append(f"- 7-class `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `casme3_recognition_manifest_motion_audit.csv`: original manifest plus motion-quality columns",
            "- `casme3_recognition_manifest_nonzero_motion.csv`: diagnostic manifest excluding zero-motion samples",
            "- `zero_motion_samples.csv`: clean samples whose motion tensor has zero magnitude",
            "- `weak_motion_samples.csv`: clean samples below the weak-motion threshold",
            "",
            "## Use",
            "",
            "Use the nonzero-motion manifest only as a diagnostic experiment. The full clean manifest remains the fair main protocol.",
        ]
    )
    (output_dir / "motion_quality_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote motion audit to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit motion tensor quality in a CAS(ME)^3 recognition manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/casme3_recognition_v2/casme3_recognition_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/casme3_motion_quality"),
    )
    parser.add_argument("--weak-mean-threshold", type=float, default=0.03)
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    audit(manifest_path=manifest_path, output_dir=output_dir, weak_mean_threshold=args.weak_mean_threshold)


if __name__ == "__main__":
    main()
