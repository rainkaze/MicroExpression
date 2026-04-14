import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import CASME2FlowDataset
from src.models.sfamnet import SFAMNetLite

# 配置与训练脚本对齐
CONFIG = {
    'processed_dir': './data/CASME II/processed',
    'csv_path': './data/CASME II/CASME2-coding-20140508.xlsx',
    'checkpoints_dir': './checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def evaluate_loso():
    all_preds = []
    all_labels = []

    # 1. 获取所有被试者 (01-26)
    all_subjects = [str(i).zfill(2) for i in range(1, 27)]

    print("开始汇总 26 个 Fold 的预测结果以进行全面检验...")

    for sub in all_subjects:
        model_path = os.path.join(CONFIG['checkpoints_dir'], f"best_model_fold_{sub}.pth")
        if not os.path.exists(model_path):
            continue

        # 加载该被试者的测试数据
        test_dataset = CASME2FlowDataset(CONFIG['processed_dir'], CONFIG['csv_path'], subjects=[sub])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 加载对应的最佳模型权重
        model = SFAMNetLite(num_classes=4).to(CONFIG['device'])
        model.load_state_dict(torch.load(model_path, map_location=CONFIG['device']))
        model.eval()

        with torch.no_grad():
            for u, v, label in test_loader:
                u, v = u.to(CONFIG['device']), v.to(CONFIG['device'])
                outputs = model(u, v)
                _, predicted = torch.max(outputs, 1)

                all_preds.append(predicted.item())
                all_labels.append(label.item())

    # 2. 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 归一化处理 (显示百分比)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 3. 可视化绘制
    plt.figure(figsize=(10, 8))
    emotion_labels = ['Positive', 'Negative', 'Surprise', 'Others']

    sns.heatmap(cm_perc, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=emotion_labels, yticklabels=emotion_labels)

    plt.title('CASME II LOSO - Confusion Matrix')
    plt.xlabel('Predicted (AI 预测)')
    plt.ylabel('Actual (人类标注)')
    plt.savefig('confusion_matrix.png')
    print("✨ 混淆矩阵已保存为 confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate_loso()