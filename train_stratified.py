import os

import numpy as np

from train_loso import CONFIG, set_seed, train_stratified


def main() -> None:
    CONFIG["protocol"] = "stratified"
    CONFIG["finetune_epochs"] = 50
    CONFIG["batch_size"] = 16

    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)
    set_seed(CONFIG["seed"])

    result = train_stratified()
    metrics = result["metrics"]

    print("\n" + "=" * 40)
    print(f"7-class stratified accuracy: {float(metrics['accuracy']) * 100:.2f}%")
    print(f"7-class stratified UAR: {float(metrics['uar']) * 100:.2f}%")
    print(f"7-class stratified Macro-F1: {float(metrics['macro_f1']) * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
