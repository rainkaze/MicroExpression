import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

from src.dataset import CASME2FlowDataset
from src.models.sfamnet import SFAMNetLite
from src.utils.logger import setup_logger

# --- 全局配置 (业界通常使用 Config 类或 YAML) ---
CONFIG = {
    'processed_dir': './data/CASME II/processed',
    'csv_path': './data/CASME II/CASME2-coding-20140508.xlsx',
    'checkpoints_dir': './checkpoints',
    'log_dir': './logs',
    'batch_size': 32,
    'epochs': 60,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


def train_one_fold(fold_id, train_subs, test_subs):
    """
    完成单一 Fold (LOSO 中的一次完整训练)
    """
    logger = setup_logger(CONFIG['log_dir'], f"fold_sub{test_subs[0]}")
    logger.info(f"开始训练 Fold: 测试集为被试者 {test_subs}")

    # 1. 准备数据加载器
    train_dataset = CASME2FlowDataset(CONFIG['processed_dir'], CONFIG['csv_path'], subjects=train_subs)
    test_dataset = CASME2FlowDataset(CONFIG['processed_dir'], CONFIG['csv_path'], subjects=test_subs)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # 2. 初始化模型、损失函数、优化器
    model = SFAMNetLite(num_classes=4).to(CONFIG['device'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_acc = 0.0

    # 3. 训练循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        for u, v, labels in train_loader:
            u, v, labels = u.to(CONFIG['device']), v.to(CONFIG['device']), labels.to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(u, v)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for u, v, labels in test_loader:
                u, v, labels = u.to(CONFIG['device']), v.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(u, v)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total if total > 0 else 0
        if acc > best_acc:
            best_acc = acc
            # 保存该 Fold 的最佳权重
            save_path = os.path.join(CONFIG['checkpoints_dir'], f"best_model_fold_{test_subs[0]}.pth")
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{CONFIG['epochs']}] Loss: {train_loss / len(train_loader):.4f} Acc: {acc:.2f}%")

    logger.info(f"Fold {test_subs[0]} 完成! 最佳准确率: {best_acc:.2f}%")
    return best_acc


def main():
    # 建立必要目录
    os.makedirs(CONFIG['checkpoints_dir'], exist_ok=True)

    # 1. 获取所有 Subject 列表 (根据 CASME II 共有 26 个被试)
    # 这里的 1-26 是 CASME II 的标准被试编号
    all_subjects = [str(i).zfill(2) for i in range(1, 27)]

    # 排除数据集中缺失的编号（如果有的话）
    # 比如某些 Subject 可能因为数据质量被剔除

    results = []

    # 2. 执行 LOSO 循环
    for i in range(len(all_subjects)):
        test_sub = [all_subjects[i]]
        train_subs = [s for s in all_subjects if s not in test_sub]

        fold_acc = train_one_fold(i, train_subs, test_sub)
        results.append(fold_acc)

    # 3. 输出最终评估结果
    print("\n" + "=" * 30)
    print(f"LOSO 最终平均准确率: {np.mean(results):.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    main()