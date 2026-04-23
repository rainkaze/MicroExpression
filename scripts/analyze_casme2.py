from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import EMOTION_TO_LABEL, normalize_emotion_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CASME II labels and processed flow files.")
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_ROOT, "data", "CASME II", "CASME2-coding-20140508.xlsx"),
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join(PROJECT_ROOT, "processed_v2"),
    )
    args = parser.parse_args()

    df = pd.read_excel(args.csv)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[pd.to_numeric(df["OnsetFrame"], errors="coerce").notnull()].copy()
    df["Subject"] = df["Subject"].apply(lambda x: str(int(x)).zfill(2))
    df["Filename"] = df["Filename"].astype(str).str.strip()
    df["emotion"] = df["Estimated Emotion"].apply(normalize_emotion_name)

    valid = []
    missing = []
    unknown = []
    for row in df.to_dict("records"):
        sample_id = f"sub{row['Subject']}_{row['Filename']}"
        emotion = row["emotion"]
        if emotion not in EMOTION_TO_LABEL:
            unknown.append((sample_id, emotion))
            continue

        u_path = os.path.join(args.processed_dir, "u", f"{sample_id}.npy")
        v_path = os.path.join(args.processed_dir, "v", f"{sample_id}.npy")
        if not os.path.exists(u_path) or not os.path.exists(v_path):
            missing.append(sample_id)
            continue
        valid.append((sample_id, emotion, row["Subject"], u_path, v_path))

    print(f"Excel rows: {len(df)}")
    print(f"Subjects: {len(df['Subject'].unique())} -> {sorted(df['Subject'].unique())}")
    print(f"Raw label distribution: {dict(Counter(df['emotion']))}")
    print(f"Valid processed samples: {len(valid)}")
    print(f"Missing processed samples: {len(missing)}")
    print(f"Unknown labels: {len(unknown)}")
    print(f"Valid label distribution: {dict(Counter(item[1] for item in valid))}")
    print(f"Valid samples per subject: {dict(sorted(Counter(item[2] for item in valid).items()))}")

    if unknown:
        print(f"First unknown labels: {unknown[:10]}")
    if missing:
        print(f"First missing samples: {missing[:10]}")

    if not valid:
        return

    stats = []
    bad_shapes = []
    for sample_id, _, _, u_path, v_path in valid:
        u = np.load(u_path).astype(np.float32)
        v = np.load(v_path).astype(np.float32)
        if u.shape != v.shape:
            bad_shapes.append((sample_id, u.shape, v.shape))
        magnitude = np.sqrt(np.square(u) + np.square(v))
        stats.append(
            [
                float(np.nanmin(u)),
                float(np.nanmax(u)),
                float(np.nanmean(u)),
                float(np.nanstd(u)),
                float(np.nanmax(magnitude)),
                float(np.nanmean(magnitude)),
                float(u.shape[0]),
                float(u.shape[1]),
            ]
        )

    arr = np.array(stats, dtype=np.float32)
    columns = "u_min u_max u_mean u_std mag_max mag_mean height width"
    print(f"Flow statistic columns: {columns}")
    print(f"Mean: {np.round(arr.mean(axis=0), 4).tolist()}")
    print(f"P95:  {np.round(np.percentile(arr, 95, axis=0), 4).tolist()}")
    print(f"Max:  {np.round(arr.max(axis=0), 4).tolist()}")
    print(f"Contains NaN: {bool(np.isnan(arr).any())}")
    if bad_shapes:
        print(f"Bad shape samples: {bad_shapes[:10]}")


if __name__ == "__main__":
    main()
