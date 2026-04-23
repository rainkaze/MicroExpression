import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.dataset import CASME2FlowDataset, EMOTION_CLASSES
from src.models.sfamnet import SFAMNetLite
from src.utils.metrics import classification_metrics


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


CONFIG = {
    "processed_dir": os.path.join(PROJECT_ROOT, "processed_v2"),
    "csv_path": os.path.join(PROJECT_ROOT, "../data", "CASME II", "CASME2-coding-20140508.xlsx"),
    "checkpoints_dir": os.path.join(PROJECT_ROOT, "../checkpoints_v2"),
    "confusion_matrix_path": os.path.join(PROJECT_ROOT, "../outputs", "confusion_matrix_loso_7class.png"),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def evaluate_loso() -> None:
    all_preds = []
    all_labels = []
    all_subjects = [str(i).zfill(2) for i in range(1, 27)]

    print("Aggregating 26 LOSO folds for the 7-class evaluation...")

    for sub in all_subjects:
        model_path = os.path.join(CONFIG["checkpoints_dir"], f"best_model_fold_{sub}.pth")
        if not os.path.exists(model_path):
            continue

        test_dataset = CASME2FlowDataset(
            CONFIG["processed_dir"],
            CONFIG["csv_path"],
            subjects=[sub],
            augment=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = SFAMNetLite(num_classes=len(EMOTION_CLASSES)).to(CONFIG["device"])
        model.load_state_dict(torch.load(model_path, map_location=CONFIG["device"]))
        model.eval()

        with torch.no_grad():
            for flow, label, _ in test_loader:
                flow = flow.to(CONFIG["device"])
                outputs = model(flow)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(label.tolist())

    if not all_labels:
        raise RuntimeError(
            f"No predictions were collected. Check checkpoint path: {CONFIG['checkpoints_dir']}"
        )

    metrics = classification_metrics(all_labels, all_preds, len(EMOTION_CLASSES))
    cm = metrics["confusion_matrix"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0)

    print(f"Accuracy: {100.0 * float(metrics['accuracy']):.2f}%")
    print(f"UAR: {100.0 * float(metrics['uar']):.2f}%")
    print(f"Macro-F1: {100.0 * float(metrics['macro_f1']):.2f}%")
    print("\nPer-class recall / F1:")
    for class_name, recall, f1 in zip(
        EMOTION_CLASSES,
        metrics["recall_per_class"],
        metrics["f1_per_class"],
    ):
        print(f"{class_name:10s} recall={100.0 * float(recall):6.2f}%  f1={100.0 * float(f1):6.2f}%")

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_perc,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=EMOTION_CLASSES,
        yticklabels=EMOTION_CLASSES,
    )
    plt.title("CASME II LOSO - 7-Class Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    os.makedirs(os.path.dirname(CONFIG["confusion_matrix_path"]), exist_ok=True)
    try:
        plt.savefig(CONFIG["confusion_matrix_path"], dpi=300)
        print(f"Saved confusion matrix to {CONFIG['confusion_matrix_path']}")
    except PermissionError:
        fallback_path = os.path.join(PROJECT_ROOT, "confusion_matrix_loso_7class_new.png")
        plt.savefig(fallback_path, dpi=300)
        print(f"Saved confusion matrix to {fallback_path}")
    plt.show()


if __name__ == "__main__":
    evaluate_loso()
