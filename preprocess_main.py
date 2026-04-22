from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocess.flow_utils import OpticalFlowEngine


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


class CASME2Preprocessor:
    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        output_dir: str,
        target_size: tuple[int, int] = (128, 128),
        apex_window: int = 2,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.target_size = target_size
        self.apex_window = max(0, int(apex_window))

        self.df = pd.read_excel(csv_path)
        self.df.columns = [str(c).strip() for c in self.df.columns]
        self.df = self.df[pd.to_numeric(self.df["OnsetFrame"], errors="coerce").notnull()].copy()
        print(f"Loaded Excel rows: {len(self.df)}")

        self.case_index = self._build_case_index()
        self.flow_engine = OpticalFlowEngine(algorithm="TV-L1")
        os.makedirs(os.path.join(output_dir, "u"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "v"), exist_ok=True)

    def _build_case_index(self) -> dict[tuple[str, str], str]:
        index: dict[tuple[str, str], str] = {}
        duplicates: dict[tuple[str, str], list[str]] = defaultdict(list)

        for current_root, dirs, files in os.walk(self.root_dir):
            subject = os.path.basename(current_root)
            match = re.fullmatch(r"sub(\d{2})", subject, flags=re.IGNORECASE)
            if not match:
                continue

            sub_id = match.group(1)
            for case_name in dirs:
                key = (sub_id, case_name)
                case_path = os.path.join(current_root, case_name)
                if key in index:
                    duplicates[key].append(case_path)
                else:
                    index[key] = case_path

        if duplicates:
            print(f"Warning: duplicate subject/case directories found: {len(duplicates)}")
        print(f"Indexed CASME II case directories: {len(index)}")
        return index

    def _get_actual_path(self, sub_id: str, filename: str) -> str | None:
        return self.case_index.get((sub_id, filename))

    @staticmethod
    def _load_img(case_path: str | None, frame_idx: int):
        if not case_path:
            return None

        patterns = [
            f"img{frame_idx}.jpg",
            f"img{str(frame_idx).zfill(3)}.jpg",
            f"img{str(frame_idx).zfill(5)}.jpg",
        ]
        for name in patterns:
            full_path = os.path.join(case_path, name)
            if os.path.exists(full_path):
                return cv2.imread(full_path)
        return None

    def _compute_window_flow(self, img_onset, case_path: str, apex: int) -> tuple[np.ndarray, np.ndarray] | None:
        flows = []
        for frame_idx in range(apex - self.apex_window, apex + self.apex_window + 1):
            if frame_idx <= 0:
                continue
            img_target = self._load_img(case_path, frame_idx)
            if img_target is None:
                continue

            img_target = cv2.resize(img_target, self.target_size)
            u, v = self.flow_engine.compute_flow(img_onset, img_target)
            flows.append((u.astype(np.float32), v.astype(np.float32)))

        if not flows:
            return None

        u = np.mean([item[0] for item in flows], axis=0)
        v = np.mean([item[1] for item in flows], axis=0)
        return u, v

    def run(self) -> None:
        success_count = 0
        missing_path = []
        missing_frame = []
        print(f"Scanning data root: {self.root_dir}")

        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                sub = str(int(row["Subject"])).zfill(2)
                filename = str(row["Filename"]).strip()
                onset = int(row["OnsetFrame"])
                apex = int(row["ApexFrame"])
                sample_id = f"sub{sub}_{filename}"

                case_path = self._get_actual_path(sub, filename)
                if case_path is None:
                    missing_path.append(sample_id)
                    continue

                img_onset = self._load_img(case_path, onset)
                if img_onset is None:
                    missing_frame.append((sample_id, onset, apex))
                    continue

                img_onset = cv2.resize(img_onset, self.target_size)
                flow = self._compute_window_flow(img_onset, case_path, apex)
                if flow is None:
                    missing_frame.append((sample_id, onset, apex))
                    continue

                u, v = flow
                np.save(os.path.join(self.output_dir, "u", f"{sample_id}.npy"), u.astype(np.float16))
                np.save(os.path.join(self.output_dir, "v", f"{sample_id}.npy"), v.astype(np.float16))
                success_count += 1
            except Exception as exc:
                missing_frame.append((str(row.get("Filename", "unknown")), "error", str(exc)))

        print("\nPreprocessing complete")
        print(f"Success: {success_count} | Failed: {len(self.df) - success_count}")
        if missing_path:
            print(f"Missing case directories ({len(missing_path)}): {missing_path[:20]}")
        if missing_frame:
            print(f"Missing frame/window entries ({len(missing_frame)}): {missing_frame[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CASME II onset-to-apex-window optical flow.")
    parser.add_argument("--root-dir", default=os.path.join(PROJECT_ROOT, "data", "CASME II", "CASME2_RAW"))
    parser.add_argument(
        "--csv-path",
        default=os.path.join(PROJECT_ROOT, "data", "CASME II", "CASME2-coding-20140508.xlsx"),
    )
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "processed_v2"))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--apex-window", type=int, default=2)
    args = parser.parse_args()

    processor = CASME2Preprocessor(
        root_dir=args.root_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        target_size=(args.size, args.size),
        apex_window=args.apex_window,
    )
    processor.run()


if __name__ == "__main__":
    main()
