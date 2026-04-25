from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.datasets import LABEL_MODES
from src.models import build_model
from src.preprocess.motion import OpticalFlowEngine, build_flow_tensor
from src.utils import load_toml_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


@dataclass(frozen=True)
class UploadedFile:
    filename: str
    content_type: str
    data: bytes


class RecognitionService:
    def __init__(self, project_root: Path, run_dir: Path, device_name: str = "auto") -> None:
        self.project_root = project_root
        self.run_dir = run_dir
        self.device = self._resolve_device(device_name)
        self.config = self._load_config()
        self.labels = LABEL_MODES[self.config["data"]["label_mode"]]
        self.model_name = self.config["model"]["name"]
        self.input_mode = self.config["data"]["input_mode"]
        self.base_channels = int(self.config["model"].get("base_channels", 32))
        self.dropout = float(self.config["model"].get("dropout", 0.2))
        self.manifest_path = self.project_root / self.config["data"]["manifest_path"]
        self.flow_engine = OpticalFlowEngine("TV-L1")
        self.models = self._load_models()
        self.samples = self._load_samples()

    def _resolve_device(self, device_name: str) -> torch.device:
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        return torch.device(device_name)

    def _load_config(self) -> dict[str, Any]:
        config_name = self.run_dir.name
        config_path = self.project_root / "configs" / "train" / "casme3" / "main" / f"{config_name}.toml"
        if not config_path.exists() and config_name.endswith("_5fold"):
            config_path = self.project_root / "configs" / "train" / "casme3" / "main" / f"{config_name.removesuffix('_5fold')}.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config for run: {self.run_dir}")
        return load_toml_config(config_path)

    def _load_models(self) -> dict[str, torch.nn.Module]:
        fold_dirs = sorted(path for path in self.run_dir.glob("fold_*") if (path / "best_model.pt").exists())
        if not fold_dirs:
            raise FileNotFoundError(f"No fold checkpoints found in {self.run_dir}")

        models: dict[str, torch.nn.Module] = {}
        for fold_dir in fold_dirs:
            model = build_model(
                model_name=self.model_name,
                num_classes=len(self.labels),
                input_mode=self.input_mode,
                base_channels=self.base_channels,
                dropout=self.dropout,
            )
            state = torch.load(fold_dir / "best_model.pt", map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            models[fold_dir.name] = model
        return models

    def metadata(self) -> dict[str, Any]:
        summary_path = self.run_dir / "summary.json"
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return {
            "run_name": self.run_dir.name,
            "device": str(self.device),
            "model_name": self.model_name,
            "input_mode": self.input_mode,
            "label_mode": self.config["data"]["label_mode"],
            "labels": self.labels,
            "folds": list(self.models.keys()),
            "summary": summary,
        }

    def sample_options(self, limit: int = 220) -> dict[str, Any]:
        return {"samples": self.samples[:limit], "total": len(self.samples)}

    def predict_sample(self, sample_id: str, fold: str) -> dict[str, Any]:
        sample = next((item for item in self.samples if item["sample_id"] == sample_id), None)
        if sample is None:
            raise ValueError(f"Unknown sample_id: {sample_id}")
        tensor_path = self.project_root / str(sample["flow_path"])
        tensor = np.load(tensor_path).astype(np.float32)
        result = self._predict_tensor(tensor[:4], fold, source="dataset_sample")
        result["sample"] = sample
        return result

    def predict_images(self, onset_bytes: bytes, apex_bytes: bytes, fold: str) -> dict[str, Any]:
        onset = self._decode_image(onset_bytes)
        apex = self._decode_image(apex_bytes)
        onset = cv2.resize(onset, (128, 128), interpolation=cv2.INTER_AREA)
        apex = cv2.resize(apex, (128, 128), interpolation=cv2.INTER_AREA)
        u, v = self.flow_engine.compute(onset, apex)
        tensor = build_flow_tensor(u, v)
        return self._predict_tensor(tensor, fold, source="computed_flow")

    def predict_npy(self, data: bytes, fold: str) -> dict[str, Any]:
        array = np.load(io.BytesIO(data)).astype(np.float32)
        if array.ndim != 3 or array.shape[0] < 4:
            raise ValueError("Expected .npy tensor shaped (4, H, W) or more channels.")
        return self._predict_tensor(array[:4], fold, source="uploaded_tensor")

    def _load_samples(self) -> list[dict[str, str]]:
        if not self.manifest_path.exists():
            return []
        samples: list[dict[str, str]] = []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("issues", "").strip():
                    continue
                label_key = "emotion_4" if self.config["data"]["label_mode"] == "4class" else "emotion_7"
                flow_path = row.get("flow_path", "")
                if not flow_path:
                    continue
                samples.append(
                    {
                        "sample_id": row.get("sample_id", ""),
                        "subject": row.get("subject", ""),
                        "video_code": row.get("video_code", ""),
                        "onset": row.get("onset", ""),
                        "apex": row.get("apex", ""),
                        "offset": row.get("offset", ""),
                        "label": row.get(label_key, ""),
                        "emotion_7": row.get("emotion_7", ""),
                        "emotion_4": row.get("emotion_4", ""),
                        "flow_path": flow_path,
                    }
                )
        return samples

    def _decode_image(self, data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _predict_tensor(self, tensor: np.ndarray, fold: str, source: str) -> dict[str, Any]:
        x = torch.from_numpy(np.ascontiguousarray(tensor.astype(np.float32))).unsqueeze(0).to(self.device)
        selected_models = self._select_models(fold)
        probabilities = []
        with torch.no_grad():
            for model in selected_models.values():
                logits = model(x)
                probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy()[0])
        mean_prob = np.mean(np.stack(probabilities, axis=0), axis=0)
        pred_idx = int(np.argmax(mean_prob))
        ranking = sorted(
            [
                {"label": label, "probability": float(prob)}
                for label, prob in zip(self.labels, mean_prob.tolist(), strict=False)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )
        return {
            "source": source,
            "fold": fold,
            "model_count": len(selected_models),
            "prediction": self.labels[pred_idx],
            "confidence": float(mean_prob[pred_idx]),
            "ranking": ranking,
            "stats": self._tensor_stats(tensor),
            "visualization": self._visualize_tensor(tensor),
        }

    def _select_models(self, fold: str) -> dict[str, torch.nn.Module]:
        if fold == "ensemble":
            return self.models
        if fold not in self.models:
            raise ValueError(f"Unknown fold: {fold}")
        return {fold: self.models[fold]}

    def _tensor_stats(self, tensor: np.ndarray) -> dict[str, float | list[int]]:
        return {
            "shape": list(tensor.shape),
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
        }

    def _visualize_tensor(self, tensor: np.ndarray) -> str:
        names = ["u", "v", "magnitude", "orientation"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for index, ax in enumerate(axes):
            ax.imshow(tensor[index], cmap="magma")
            ax.set_title(names[index])
            ax.axis("off")
        fig.tight_layout(pad=0.4)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_multipart(content_type: str, body: bytes) -> dict[str, UploadedFile | str]:
    boundary_token = "boundary="
    if boundary_token not in content_type:
        raise ValueError("Missing multipart boundary.")
    boundary = content_type.split(boundary_token, 1)[1].strip().strip('"').encode("utf-8")
    result: dict[str, UploadedFile | str] = {}
    for raw_part in body.split(b"--" + boundary):
        part = raw_part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, data = part.split(b"\r\n\r\n", 1)
        headers = raw_headers.decode("utf-8", errors="replace").split("\r\n")
        disposition = next((line for line in headers if line.lower().startswith("content-disposition:")), "")
        content_type_header = next((line for line in headers if line.lower().startswith("content-type:")), "")
        attrs = parse_content_disposition(disposition)
        name = attrs.get("name")
        if not name:
            continue
        data = data.removesuffix(b"\r\n")
        filename = attrs.get("filename")
        if filename:
            file_type = content_type_header.split(":", 1)[1].strip() if ":" in content_type_header else ""
            result[name] = UploadedFile(filename=filename, content_type=file_type, data=data)
        else:
            result[name] = data.decode("utf-8", errors="replace")
    return result


def parse_content_disposition(value: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in value.split(";"):
        item = item.strip()
        if "=" in item:
            key, raw = item.split("=", 1)
            attrs[key.lower()] = raw.strip().strip('"')
    return attrs


class RecognitionHandler(BaseHTTPRequestHandler):
    service: RecognitionService

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/metadata":
            self.write_json(self.service.metadata())
            return
        if parsed.path == "/api/samples":
            self.write_json(self.service.sample_options())
            return
        path = "index.html" if parsed.path == "/" else parsed.path.lstrip("/")
        static_path = (STATIC_ROOT / path).resolve()
        if not str(static_path).startswith(str(STATIC_ROOT.resolve())) or not static_path.exists():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(static_path.name)[0] or "application/octet-stream"
        data = static_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            fields = parse_multipart(content_type, body)
            fold = str(fields.get("fold", "ensemble"))
            if parsed.path == "/api/predict-images":
                onset = fields.get("onset")
                apex = fields.get("apex")
                if not isinstance(onset, UploadedFile) or not isinstance(apex, UploadedFile):
                    raise ValueError("Upload both onset and apex images.")
                self.write_json(self.service.predict_images(onset.data, apex.data, fold))
                return
            if parsed.path == "/api/predict-tensor":
                tensor = fields.get("tensor")
                if not isinstance(tensor, UploadedFile):
                    raise ValueError("Upload a .npy tensor.")
                self.write_json(self.service.predict_npy(tensor.data, fold))
                return
            if parsed.path == "/api/predict-sample":
                sample_id = fields.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("Choose a dataset sample.")
                self.write_json(self.service.predict_sample(sample_id, fold))
                return
            self.send_error(404)
        except Exception as exc:
            self.write_json({"error": str(exc)}, status=400)

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[recognition-web] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local micro-expression recognition web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs_dev2" / "flow4_main_balanced_softmax_5fold",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    service = RecognitionService(PROJECT_ROOT, args.run_dir, args.device)
    RecognitionHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RecognitionHandler)
    print(f"Recognition web app: http://{args.host}:{args.port}")
    print(f"Run: {args.run_dir}")
    print(f"Device: {service.device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
