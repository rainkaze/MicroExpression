import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import AttentionNet
import os


def predict(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 类别映射 (根据 dataset.py 输出的结果填写)
    class_map = {
        0: 'disgust', 1: 'fear', 2: 'happiness', 3: 'others',
        4: 'repression', 5: 'sadness', 6: 'surprise'
    }

    # 2. 初始化模型并加载权重
    model = AttentionNet(num_classes=len(class_map)).to(device)
    model_path = './checkpoints/best_model.pth'

    if not os.path.exists(model_path):
        print("❌ 未找到训练好的模型权重文件！请先运行 train.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 4. 读取图片
    try:
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ 读取图片失败: {e}")
        return

    # 5. 推理
    with torch.no_grad():
        output = model(input_tensor)
        # 获取概率
        probabilities = F.softmax(output, dim=1)
        prob, pred = torch.max(probabilities, 1)

    print("-" * 30)
    print(f"📸 目标图片: {img_path}")
    print(f"🔍 预测类别: {class_map[pred.item()]}")
    print(f"📈 置信度: {prob.item() * 100:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    # 在这里输入你想测试的照片路径
    # 比如你在项目根目录下放了一张 me.jpg
    predict('me.jpg')