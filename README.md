# Micro-expression Recognition Based on Attention Network 
基于注意力网络的微表情识别

## 🚀 项目简介
本项目旨在利用深度学习技术识别面部微表情。通过在卷积神经网络（CNN）中引入**空间注意力机制（Spatial Attention）**，模型能够更精准地聚焦于面部细微的肌肉运动区域。

## 🛠️ 技术栈
- **硬件**: NVIDIA GeForce RTX 5070 (8GB)
- **框架**: PyTorch, Torchvision
- **数据集**: Kaggle 微表情图像数据集（约 17,000 张照片）

## 📊 实验结果
在平衡模式下，模型经过 10 轮迭代达到以下表现：
- **验证集准确率**: 98.27%
- **训练耗时**: 约 28s/Epoch
- **散热表现**: GPU 维持在 50°C 左右，CPU 维持在 88°C 以下

## 📂 目录结构
- `dataset.py`: 数据预处理与加载流水线
- `model.py`: 空间注意力网络架构定义
- `train.py`: 训练循环与模型保存逻辑
- `predict.py`: 单张测试图片推理脚本

## 🔧 使用说明
1. 安装依赖: `pip install -r requirements.txt`
2. 开始训练: `python train.py`
3. 预测照片: `python predict.py`