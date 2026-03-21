import torch
import matplotlib.pyplot as plt
from model import AttentionNet
from dataset import get_data_loader
from train import train_model

# === GPU 加速 ===
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=" * 60)
    print("      基于注意力网络的微表情识别（7分类·高精度版）")
    print("=" * 60)

    # 显示GPU信息
    if torch.cuda.is_available():
        print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"✅ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ 使用 CPU 训练")

    print("\n加载数据集...")
    loader, classes = get_data_loader(batch_size=128)
    print("分类类别:", classes)

    print("\n构建注意力网络...")
    model = AttentionNet(num_classes=7)

    print("\n开始训练...")
    history = train_model(model, loader, epochs=25)

    # ======================
    # 训练结果可视化（汇报用）
    # ======================
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["acc"], label="准确率")
    plt.title("训练准确率")
    plt.xlabel("Epoch")
    plt.ylabel("Acc (%)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="损失", color="orange")
    plt.title("训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_result.png")
    plt.show()

    print("\n🎉 可视化图片已保存：training_result.png")