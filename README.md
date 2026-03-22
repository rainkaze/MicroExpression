# Micro-expression Recognition Based on Attention Network 
基于注意力网络的微表情识别
## 🚀 项目简介
本项目实现了一种基于计算机视觉的微表情识别系统。不同于传统的静态图像分类，本项目通过计算连续帧间的 **TV-L1 光流场** 来捕捉面部极其细微的肌肉运动，并利用 **SFAMNet-Lite** 网络提取时空特征。

项目核心采用 **LOSO (Leave-One-Subject-Out)** 留一受试者交叉验证法，确保了模型对不同个体面部特征的泛化能力。

## 📂 目录结构
```text
MicroExpression/
├── data/                  # 数据集存放目录 (已忽略)
│   └── CASME II/          # 原始数据集与标签 Excel
├── src/                   # 核心源代码
│   ├── models/            # 网络模型定义 (SFAMNet-Lite)
│   ├── preprocess/        # 预处理工具 (TV-L1 光流计算)
│   ├── utils/             # 工具函数 (日志、路径搜索)
│   └── dataset.py         # 针对光流数据的 PyTorch Dataset 实现
├── logs/                  # 训练日志记录
├── checkpoints/           # 训练最优模型权重 (.pth)
├── preprocess_main.py     # 预处理启动脚本 (生成 .npy 光流数据)
├── train_loso.py          # LOSO 训练启动脚本
├── requirements.txt       # 环境依赖清单
└── LICENSE                # MIT 开源协议
```

## 📊 实验表现 (CASME II 数据集)

  - **验证方法**: Leave-One-Subject-Out (LOSO)
  - **分类标准**: 4 分类 (Positive, Negative, Surprise, Others)
  - **预处理**: 128x128 像素, TV-L1 稠密光流
  - **最终平均准确率**: **67.99%**
  - **硬件环境**: NVIDIA GeForce RTX 5070 (8GB)

## 🛠️ 环境配置

```bash
# 创建环境
conda create -n micro_expression python=3.11
conda activate micro_expression

# 安装依赖
pip install -r requirements.txt
```

## 🔧 使用说明

### 1\. 数据准备

将 CASME II 数据集解压至 `data` 目录下，确保包含 `CASME2_RAW` 文件夹和标注 Excel 文件。

### 2\. 预处理 (计算光流)

```bash
python preprocess_main.py
```

该脚本会自动扫描被试者文件夹，计算起始帧与最高潮帧间的光流，并保存为 `.npy` 格式。

### 3\. 开始训练 (LOSO)

```bash
python train_loso.py
```

训练完成后，`checkpoints/` 目录下会生成 26 个针对不同被试者的最优模型。

## ⚖️ 开源协议

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。

````

