import torch
import numpy as np
from src.models.sfamnet import SFAMNetLite


def quick_predict(u_path, v_path, model_fold='08'):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SFAMNetLite(num_classes=4).to(device)

    weight_path = f'./checkpoints/best_model_fold_{model_fold}.pth'
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    # 读取预处理好的光流数据
    u = np.load(u_path).astype(np.float32)
    v = np.load(v_path).astype(np.float32)

    u_tensor = torch.from_numpy(u).unsqueeze(0).unsqueeze(0).to(device)
    v_tensor = torch.from_numpy(v).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(u_tensor, v_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    emotions = ['Positive', 'Negative', 'Surprise', 'Others']
    print(f"--- 预测完成 ---")
    print(f"AI 认为该微表情是: {emotions[pred]} (置信度: {probs[0][pred]:.2%})")


if __name__ == "__main__":
    test_u = './data/CASME II/processed/u/sub01_EP02_01f.npy'
    test_v = './data/CASME II/processed/v/sub01_EP02_01f.npy'
    quick_predict(test_u, test_v)