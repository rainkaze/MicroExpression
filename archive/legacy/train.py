import torch
import torch.nn as nn
import torch.optim as optim
from model import AttentionNet
from dataset import get_dataloaders
from tqdm import tqdm
import matplotlib.pyplot as plt  # 用于绘图
import os


def main():
    # --- 1. 参数配置 ---
    DATA_PATH = r'C:\data'
    CHECKPOINT_DIR = '../checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10  # 你可以根据需要改回 30
    lr = 0.0001

    # --- 2. 准备数据和模型 ---
    train_loader, val_loader, class_idx = get_dataloaders(DATA_PATH)
    model = AttentionNet(num_classes=len(class_idx)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 用于记录绘图数据
    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": []
    }

    best_val_acc = 0.0

    # --- 3. 训练循环 ---
    print(f"🚀 开始训练...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct, train_total, running_loss = 0, 0, 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]")

        for imgs, labels in train_loop:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loop.set_postfix(acc=f"{100. * train_correct / train_total:.2f}%")

        # 计算本轮平均数据
        epoch_train_acc = 100. * train_correct / train_total
        epoch_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100. * val_correct / val_total

        # 记录到 history
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(epoch_loss)

        print(f"📊 Epoch {epoch} - Train Acc: {epoch_train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

    # --- 4. 训练结束后的可视化 ---
    draw_result(history)


def draw_result(history):
    plt.figure(figsize=(12, 5))

    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label='Train Accuracy', color='#1f77b4', marker='o')
    plt.plot(history["val_acc"], label='Val Accuracy', color='#ff7f0e', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label='Train Loss', color='#d62728', linestyle='--')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('./confusion_matrices/training_result.png')  # 自动保存结果图
    print("\n📈 训练曲线图已保存至项目根目录: training_result.png")
    plt.show()  # 弹出显示


if __name__ == "__main__":
    main()