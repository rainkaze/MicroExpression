import numpy as np
import torch

from src.dataset import EMOTION_CLASSES
from src.models.sfamnet import SFAMNetLite


def quick_predict(u_path: str, v_path: str, model_fold: str = "08") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFAMNetLite(num_classes=len(EMOTION_CLASSES)).to(device)

    weight_path = f"./checkpoints/7class/best_model_fold_{model_fold}.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    u = np.load(u_path).astype(np.float32)
    v = np.load(v_path).astype(np.float32)

    u_tensor = torch.from_numpy(u).unsqueeze(0).unsqueeze(0).to(device)
    v_tensor = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(u_tensor, v_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print("--- Prediction Complete ---")
    print(f"Predicted emotion: {EMOTION_CLASSES[pred]} (confidence: {probs[0][pred]:.2%})")


if __name__ == "__main__":
    test_u = "./data/CASME II/processed/u/sub01_EP02_01f.npy"
    test_v = "./data/CASME II/processed/v/sub01_EP02_01f.npy"
    quick_predict(test_u, test_v)
