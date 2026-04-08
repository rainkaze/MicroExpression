import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.models.sfamnet import SFAMNetLite
from src.preprocess.flow_utils import OpticalFlowEngine

CLASS_NAMES = ["positive", "negative", "surprise", "others"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use a trained checkpoint to predict micro-expression from two images or a video.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth).")
    parser.add_argument("--image-a", help="Path to onset image.")
    parser.add_argument("--image-b", help="Path to apex image.")
    parser.add_argument("--video", help="Path to custom video.")
    parser.add_argument("--onset-frame", type=int, default=0, help="Onset frame index when using --video.")
    parser.add_argument("--apex-frame", type=int, default=1, help="Apex frame index when using --video.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="./outputs/inference")
    parser.add_argument("--target-size", type=int, default=128)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> SFAMNetLite:
    model = SFAMNetLite(num_classes=len(CLASS_NAMES)).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to read frame {frame_idx} from video: {video_path}")
    return frame


def make_color_flow(flow_u: np.ndarray, flow_v: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow_u, flow_v, angleInDegrees=True)
    hsv = np.zeros((flow_u.shape[0], flow_u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8(angle / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_visualizations(output_dir: str, image_a: np.ndarray, image_b: np.ndarray, flow_u: np.ndarray, flow_v: np.ndarray) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    onset_path = os.path.join(output_dir, "onset.png")
    apex_path = os.path.join(output_dir, "apex.png")
    flow_vis_path = os.path.join(output_dir, "flow_visualization.png")
    uv_path = os.path.join(output_dir, "flow_components.npz")

    cv2.imwrite(onset_path, image_a)
    cv2.imwrite(apex_path, image_b)
    cv2.imwrite(flow_vis_path, make_color_flow(flow_u, flow_v))
    np.savez_compressed(uv_path, u=flow_u.astype(np.float32), v=flow_v.astype(np.float32))

    return {
        "onset": onset_path,
        "apex": apex_path,
        "flow_visualization": flow_vis_path,
        "flow_components": uv_path,
    }


def predict_probabilities(model: SFAMNetLite, flow_u: np.ndarray, flow_v: np.ndarray, device: str) -> List[Tuple[str, float]]:
    u_tensor = torch.from_numpy(flow_u.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    v_tensor = torch.from_numpy(flow_v.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(u_tensor, v_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return sorted(
        [(class_name, float(prob)) for class_name, prob in zip(CLASS_NAMES, probabilities)],
        key=lambda item: item[1],
        reverse=True,
    )


def prepare_images(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    if args.image_a and args.image_b:
        return read_image(args.image_a), read_image(args.image_b)
    if args.video:
        return read_video_frame(args.video, args.onset_frame), read_video_frame(args.video, args.apex_frame)
    raise ValueError("Please provide either --image-a/--image-b or --video.")


def main() -> None:
    args = parse_args()
    image_a, image_b = prepare_images(args)
    image_a = cv2.resize(image_a, (args.target_size, args.target_size))
    image_b = cv2.resize(image_b, (args.target_size, args.target_size))

    flow_engine = OpticalFlowEngine(algorithm="TV-L1")
    flow_u, flow_v = flow_engine.compute_flow(image_a, image_b)

    model = load_model(args.checkpoint, args.device)
    ranked_predictions = predict_probabilities(model, flow_u, flow_v, args.device)
    artifact_paths = save_visualizations(args.output_dir, image_a, image_b, flow_u, flow_v)

    print("Inference complete. Ranked predictions:")
    for class_name, probability in ranked_predictions:
        print(f"  - {class_name:>8}: {probability * 100:.2f}%")

    print("\nSaved artifacts:")
    for name, path in artifact_paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
