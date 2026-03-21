import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import AttentionNet
import os


def predict(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 类别映射 (请确保顺序与训练时 datasets.ImageFolder 识别的一致)
    class_map = {
        0: 'disgust', 1: 'fear', 2: 'happiness', 3: 'others',
        4: 'repression', 5: 'sadness', 6: 'surprise'
    }

    # 2. 加载模型
    model = AttentionNet(num_classes=len(class_map)).to(device)
    model_path = './checkpoints/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 图像预处理 (加入了 CenterCrop 解决压扁问题)
    transform = transforms.Compose([
        transforms.Resize(128),  # 缩放短边
        transforms.CenterCrop((128, 128)),  # 居中裁剪，保持纵横比
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 4. 推理
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        # 获取前 3 个最高概率的类别索引和数值
        topk_probs, topk_indices = torch.topk(probabilities, 3)

    print("-" * 35)
    print(f"📸 目标图片: {img_path}")
    print("🔍 模型预测可能性排序：")
    for i in range(3):
        label = class_map[topk_indices[i].item()]
        prob = topk_probs[i].item() * 100
        print(f"   Top {i + 1}: {label:<10} | 置信度: {prob:.2f}%")
    print("-" * 35)


if __name__ == "__main__":
    # 在这里输入你想测试的照片路径
    # 比如你在项目根目录下放了一张 me.jpg
    predict('predictPic/pre06.jpg')