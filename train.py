import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, loader, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

    history = {
        "loss": [],
        "acc": []
    }

    print(f"\n✅ 使用设备: {device}")
    print(f"✅ 开始训练 7 分类微表情识别\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for img, label in loop:
            img, label = img.to(device), label.to(device)

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            loop.set_postfix(
                loss=total_loss/(total+1),
                acc=100*correct/total
            )

        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / total
        history["loss"].append(avg_loss)
        history["acc"].append(avg_acc)

        print(f"✅ Epoch {epoch+1} 准确率: {avg_acc:.2f}%")

    torch.save(model.state_dict(), "micro_expression_7class.pth")
    print("\n🎉 训练完成！模型已保存")
    return history