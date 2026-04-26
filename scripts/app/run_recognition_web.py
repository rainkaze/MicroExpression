from __future__ import annotations

import argparse
import base64
import cgi
import io
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import LABEL_MODES
from src.models import build_model


STATIC_ROOT = PROJECT_ROOT / "apps" / "recognition_web" / "static"
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "runs"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "casme3_scene_flow" / "casme3_scene_flow_manifest.csv"


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def normalize_path(value: str) -> Path:
    return PROJECT_ROOT / value.replace("\\", "/")


def to_float_tensor(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"需要形状为 (C,H,W) 或 (H,W) 的 .npy，当前形状为 {array.shape}")
    return np.clip(array, -1.0, 1.0)


def adapt_input(array: np.ndarray, input_mode: str) -> np.ndarray:
    array = to_float_tensor(array)
    channels = array.shape[0]
    if input_mode == "uvd":
        if channels != 3:
            raise ValueError("该模型需要 3 通道 UVD 输入，上传文件应为 (3,H,W)。")
        return array
    if input_mode == "uv":
        if channels < 2:
            raise ValueError("该模型需要 UV 输入，上传文件至少需要 2 个通道。")
        return array[:2]
    if input_mode == "depth":
        if channels == 1:
            return array[:1]
        if channels >= 3:
            return array[2:3]
        raise ValueError("该模型需要 depth 输入，上传 UV 文件无法推断 depth。")
    raise ValueError(f"未知 input_mode: {input_mode}")


def signed_image(channel: np.ndarray) -> str:
    channel = np.asarray(channel, dtype=np.float32)
    scale = float(np.percentile(np.abs(channel), 98))
    if scale < 1e-6:
        scale = 1.0
    x = np.clip(channel / scale, -1.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    positive = x >= 0
    rgb[..., 0] = np.where(positive, 255, (1.0 + x) * 255).astype(np.uint8)
    rgb[..., 1] = (255 * (1.0 - np.abs(x) * 0.55)).astype(np.uint8)
    rgb[..., 2] = np.where(positive, (1.0 - x) * 255, 255).astype(np.uint8)
    return image_to_data_url(rgb)


def heat_image(channel: np.ndarray) -> str:
    channel = np.asarray(channel, dtype=np.float32)
    high = float(np.percentile(channel, 98))
    low = float(np.percentile(channel, 2))
    if high - low < 1e-6:
        high = low + 1.0
    x = np.clip((channel - low) / (high - low), 0.0, 1.0)
    rgb = np.zeros((*x.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (255 * np.clip(1.8 * x, 0, 1)).astype(np.uint8)
    rgb[..., 1] = (255 * np.clip(1.8 * (1 - np.abs(x - 0.55)), 0, 1)).astype(np.uint8)
    rgb[..., 2] = (255 * np.clip(1.5 * (1 - x), 0, 1)).astype(np.uint8)
    return image_to_data_url(rgb)


def image_to_data_url(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb).resize((192, 192), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def channel_stats(array: np.ndarray) -> list[dict]:
    names = ["u", "v", "depth_delta"] if array.shape[0] == 3 else ["u", "v"] if array.shape[0] == 2 else ["depth_delta"]
    rows = []
    for index, name in enumerate(names):
        channel = array[index]
        rows.append(
            {
                "name": name,
                "min": float(channel.min()),
                "max": float(channel.max()),
                "mean": float(channel.mean()),
                "std": float(channel.std()),
                "abs_mean": float(np.mean(np.abs(channel))),
            }
        )
    return rows


def visual_payload(array: np.ndarray) -> dict:
    array = to_float_tensor(array)
    images = []
    names = ["u", "v", "depth_delta"] if array.shape[0] == 3 else ["u", "v"] if array.shape[0] == 2 else ["depth_delta"]
    for index, name in enumerate(names):
        images.append({"name": name, "src": signed_image(array[index])})
    if array.shape[0] >= 2:
        magnitude = np.sqrt(array[0] ** 2 + array[1] ** 2)
        images.append({"name": "flow_magnitude", "src": heat_image(magnitude)})
    if array.shape[0] >= 3:
        images.append({"name": "depth_abs", "src": heat_image(np.abs(array[2]))})
    return {"channels": images, "stats": channel_stats(array)}


class ModelRegistry:
    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._cache: dict[str, list[torch.nn.Module]] = {}

    def discover(self) -> list[dict]:
        models = []
        for summary_path in sorted(RUNS_ROOT.glob("*/summary.json")):
            run_root = summary_path.parent
            result_path = run_root / "fold_0" / "result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoints = sorted(run_root.glob("fold_*/best_model.pt"))
            if not checkpoints:
                continue
            config = result["config"]
            models.append(
                {
                    "run_name": run_root.name,
                    "label_mode": result["label_mode"],
                    "input_mode": result["input_mode"],
                    "labels": result["labels"],
                    "model_name": config["model"]["name"],
                    "base_channels": int(config["model"].get("base_channels", 32)),
                    "dropout": float(config["model"].get("dropout", 0.25)),
                    "folds": len(checkpoints),
                    "summary": json.loads(summary_path.read_text(encoding="utf-8")),
                }
            )
        return models

    def get_meta(self, run_name: str) -> dict:
        for item in self.discover():
            if item["run_name"] == run_name:
                return item
        raise KeyError(f"未找到模型 run: {run_name}")

    def load_ensemble(self, run_name: str) -> list[torch.nn.Module]:
        if run_name in self._cache:
            return self._cache[run_name]
        meta = self.get_meta(run_name)
        models = []
        for checkpoint in sorted((RUNS_ROOT / run_name).glob("fold_*/best_model.pt")):
            model = build_model(
                meta["model_name"],
                num_classes=len(meta["labels"]),
                input_mode=meta["input_mode"],
                base_channels=meta["base_channels"],
                dropout=meta["dropout"],
            )
            state = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            models.append(model)
        if not models:
            raise RuntimeError(f"模型没有可用 checkpoint: {run_name}")
        self._cache[run_name] = models
        return models

    @torch.no_grad()
    def predict(self, run_name: str, source_array: np.ndarray) -> dict:
        meta = self.get_meta(run_name)
        array = adapt_input(source_array, meta["input_mode"])
        tensor = torch.from_numpy(array[None]).float().to(self.device)
        probs = []
        for model in self.load_ensemble(run_name):
            logits = model(tensor)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy()[0])
        mean_prob = np.mean(np.stack(probs, axis=0), axis=0)
        top_index = int(np.argmax(mean_prob))
        return {
            "run_name": run_name,
            "label_mode": meta["label_mode"],
            "input_mode": meta["input_mode"],
            "predicted_label": meta["labels"][top_index],
            "confidence": float(mean_prob[top_index]),
            "probabilities": [
                {"label": label, "probability": float(mean_prob[index])}
                for index, label in enumerate(meta["labels"])
            ],
        }


class DataStore:
    def __init__(self) -> None:
        self.df = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
        if not self.df.empty:
            issue_column = "recognition_issues" if "recognition_issues" in self.df.columns else "issues"
            self.df = self.df[self.df[issue_column].fillna("").astype(str) == ""].reset_index(drop=True)

    def sample_rows(self, limit: int = 80, query: str = "", label_mode: str = "4class") -> list[dict]:
        if self.df.empty:
            return []
        label_col = "emotion_4" if label_mode == "4class" else "emotion_7"
        df = self.df
        if query:
            q = query.lower()
            mask = (
                df["sample_id"].astype(str).str.lower().str.contains(q, regex=False)
                | df["subject"].astype(str).str.lower().str.contains(q, regex=False)
                | df[label_col].astype(str).str.lower().str.contains(q, regex=False)
            )
            df = df[mask]
        return [
            {
                "sample_id": str(row["sample_id"]),
                "subject": str(row["subject"]),
                "emotion_4": str(row["emotion_4"]),
                "emotion_7": str(row["emotion_7"]),
            }
            for row in df.head(limit).to_dict("records")
        ]

    def load_sample(self, sample_id: str) -> tuple[np.ndarray, dict]:
        matches = self.df[self.df["sample_id"].astype(str) == sample_id]
        if matches.empty:
            raise KeyError(f"未找到样本: {sample_id}")
        row = matches.iloc[0].to_dict()
        array = np.load(normalize_path(str(row["uvd_path"]))).astype(np.float32)
        meta = {
            "sample_id": str(row["sample_id"]),
            "subject": str(row["subject"]),
            "emotion_4": str(row["emotion_4"]),
            "emotion_7": str(row["emotion_7"]),
            "au": str(row.get("au", "")),
            "onset": int(row["onset"]),
            "apex": int(row["apex"]),
        }
        return array, meta


registry = ModelRegistry()
store = DataStore()


class RecognitionHandler(BaseHTTPRequestHandler):
    server_version = "MicroExpressionRecognitionWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/models":
            json_response(self, {"models": registry.discover(), "device": str(registry.device)})
            return
        if parsed.path == "/api/samples":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            label_mode = params.get("label_mode", ["4class"])[0]
            json_response(self, {"samples": store.sample_rows(query=query, label_mode=label_mode)})
            return
        if parsed.path == "/api/predict":
            self.handle_predict(parsed)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict_upload":
            self.handle_predict_upload()
            return
        json_response(self, {"error": "未知接口"}, status=404)

    def handle_predict(self, parsed) -> None:
        try:
            params = parse_qs(parsed.query)
            sample_id = params.get("sample_id", [""])[0]
            run_names = [item for item in params.get("models", [""])[0].split(",") if item]
            if not sample_id or not run_names:
                raise ValueError("需要 sample_id 和 models 参数。")
            array, sample_meta = store.load_sample(sample_id)
            predictions = [registry.predict(run_name, array) for run_name in run_names]
            json_response(self, {"sample": sample_meta, "predictions": predictions, "visual": visual_payload(array)})
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=400)

    def handle_predict_upload(self) -> None:
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )
            run_names = [item for item in form.getfirst("models", "").split(",") if item]
            if not run_names:
                raise ValueError("请至少选择一个模型。")
            file_item = form["file"] if "file" in form else None
            if file_item is None or not getattr(file_item, "file", None):
                raise ValueError("请上传 .npy 文件。")
            data = file_item.file.read()
            array = np.load(io.BytesIO(data)).astype(np.float32)
            predictions = [registry.predict(run_name, array) for run_name in run_names]
            json_response(
                self,
                {
                    "sample": {"sample_id": getattr(file_item, "filename", "uploaded.npy"), "source": "uploaded"},
                    "predictions": predictions,
                    "visual": visual_payload(array),
                },
            )
        except Exception as exc:
            json_response(self, {"error": str(exc)}, status=400)

    def serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            file_path = STATIC_ROOT / "index.html"
        else:
            file_path = STATIC_ROOT / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            json_response(self, {"error": "文件不存在"}, status=404)
            return
        content = file_path.read_bytes()
        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        print(f"[web] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local CAS(ME)^3 recognition demo web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    global registry
    registry = ModelRegistry(device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), RecognitionHandler)
    print(f"Recognition web app: http://{args.host}:{args.port}")
    print(f"Device: {registry.device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
