import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import CASME2FlowDataset
from src.models.sfamnet import SFAMNetLite
from src.utils.logger import setup_logger
from src.utils.metrics import CLASS_NAMES, classification_report

# --- 全局配置 ---
CONFIG = {
    'processed_dir': './data/CASME II/processed',
    'csv_path': './data/CASME II/CASME2-coding-20140508.xlsx',
    'checkpoints_dir': './checkpoints',
    'log_dir': './logs',
    'plots_dir': './outputs/training',
    'batch_size': 32,
    'epochs': 60,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def save_history_plot(history: Dict[str, List[float]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train_loss', color='#d62728')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='train_acc', color='#1f77b4')
    plt.plot(history['test_acc'], label='test_acc', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Fold Accuracy')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, object]:
    y_true: List[int] = []
    y_pred: List[int] = []

    model.eval()
    with torch.no_grad():
        for u, v, labels in dataloader:
            u = u.to(device)
            v = v.to(device)
            labels = labels.to(device)
            outputs = model(u, v)
            predicted = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    return classification_report(y_true, y_pred, class_names=CLASS_NAMES)


def train_one_fold(fold_id, train_subs, test_subs):
    logger = setup_logger(CONFIG['log_dir'], f"fold_sub{test_subs[0]}")
    logger.info(f"开始训练 Fold: 测试集为被试者 {test_subs}")

    train_dataset = CASME2FlowDataset(CONFIG['processed_dir'], CONFIG['csv_path'], subjects=train_subs)
    test_dataset = CASME2FlowDataset(CONFIG['processed_dir'], CONFIG['csv_path'], subjects=test_subs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )

    model = SFAMNetLite(num_classes=4).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_acc = 0.0
    best_report = None
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for u, v, labels in train_loader:
            u = u.to(CONFIG['device'])
            v = v.to(CONFIG['device'])
            labels = labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(u, v)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss_avg = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0.0
        report = evaluate_model(model, test_loader, CONFIG['device'])
        acc = report['accuracy'] * 100

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_report = report
            save_path = os.path.join(CONFIG['checkpoints_dir'], f"best_model_fold_{test_subs[0]}.pth")
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == CONFIG['epochs'] - 1:
            logger.info(
                "Epoch [%s/%s] Loss: %.4f Train Acc: %.2f%% Test Acc: %.2f%% UAR: %.4f UF1: %.4f",
                epoch + 1,
                CONFIG['epochs'],
                train_loss_avg,
                train_acc,
                acc,
                report['uar'],
                report['uf1'],
            )

    plot_path = os.path.join(CONFIG['plots_dir'], f"fold_sub{test_subs[0]}.png")
    save_history_plot(history, plot_path)

    metrics_path = os.path.join(CONFIG['plots_dir'], f"fold_sub{test_subs[0]}_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as file:
        serializable_report = dict(best_report)
        serializable_report['confusion_matrix'] = best_report['confusion_matrix'].tolist() if best_report else []
        json.dump(
            {
                'fold': test_subs[0],
                'best_accuracy': best_acc,
                'history': history,
                'metrics': serializable_report,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Fold {test_subs[0]} 完成! 最佳准确率: {best_acc:.2f}%")
    return best_acc, best_report



def main():
    os.makedirs(CONFIG['checkpoints_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)

    all_subjects = [str(i).zfill(2) for i in range(1, 27)]
    results = []
    reports = []

    for i in range(len(all_subjects)):
        test_sub = [all_subjects[i]]
        train_subs = [s for s in all_subjects if s not in test_sub]
        fold_acc, fold_report = train_one_fold(i, train_subs, test_sub)
        results.append(fold_acc)
        reports.append(fold_report)

    mean_acc = float(np.mean(results)) if results else 0.0
    mean_uar = float(np.mean([report['uar'] for report in reports if report])) if reports else 0.0
    mean_uf1 = float(np.mean([report['uf1'] for report in reports if report])) if reports else 0.0

    summary_path = os.path.join(CONFIG['plots_dir'], 'loso_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(
            {
                'fold_accuracies': results,
                'mean_accuracy': mean_acc,
                'mean_uar': mean_uar,
                'mean_uf1': mean_uf1,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 30)
    print(f"LOSO 最终平均准确率: {mean_acc:.2f}%")
    print(f"LOSO 最终平均 UAR: {mean_uar:.4f}")
    print(f"LOSO 最终平均 UF1: {mean_uf1:.4f}")
    print(f"训练曲线与指标已保存到: {CONFIG['plots_dir']}")
    print("=" * 30)


if __name__ == "__main__":
    main()
