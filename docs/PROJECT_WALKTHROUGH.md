# MicroExpression 项目拆解说明

本文档面向第一次接触微表情识别、光流、注意力模块的同学，帮助快速看懂当前仓库的真实工作流、问题和后续改进方向。

## 1. 当前项目到底在做什么

一句话概括：

> 先从 CASME II 标注中找到每个样本的 onset（开始帧）和 apex（峰值帧），计算两帧之间的光流，再把光流送入带 CBAM 注意力模块的双分支卷积网络做 4 分类识别。

当前主线文件：

- 预处理：`preprocess_main.py`
- 数据集：`src/dataset.py`
- 模型：`src/models/sfamnet.py`
- 注意力模块：`src/models/blocks.py`
- 训练：`train_loso.py`
- 自定义推理：`infer_custom.py`

## 2. 数据集内容与分类

### 2.1 当前使用的数据集

当前仓库默认使用 **CASME II**，路径约定为：

- 原始图像目录：`data/CASME II/CASME2_RAW`
- 标注 Excel：`data/CASME II/CASME2-coding-20140508.xlsx`

### 2.2 当前分类策略

代码里实际不是做 7 分类，而是把原始情绪标签合并成 **4 分类**：

- positive：`happiness`、`repressed happiness`
- negative：`disgust`、`sadness`、`fear`
- surprise：`surprise`
- others：`others`、`repression`、`tense`

这种合并可以缓解样本过少的问题，但会损失标签细粒度信息。

### 2.3 CASME II 是否合适

**合适，但不够。**

它适合作为本科毕设的起点，因为：

- 是微表情领域常见公开数据；
- 有 onset/apex 标注，便于快速搭建光流方法；
- LOSO 评估方便做跨被试实验。

它的问题也非常明显：

- 样本量小；
- 类别不均衡；
- 跨人泛化难；
- 很容易过拟合。

### 2.4 更合适或可联合的数据集

如果后续条件允许，更建议考虑：

- SAMM
- SMIC
- CAS(ME)^2

更现实的路线通常是：

1. 先在 CASME II 上把流程跑通；
2. 再尝试跨库训练/测试；
3. 再考虑把多个数据集合并后做统一标签映射。

## 3. 预处理方法：合理吗？

### 3.1 当前做法

当前代码的预处理流程是：

1. 读取 Excel；
2. 找到 `Subject`、`Filename`、`OnsetFrame`、`ApexFrame`；
3. 读取 onset/apex 两张图；
4. resize 到 128×128；
5. 计算 TV-L1 光流；
6. 保存 `u`、`v` 两个 `.npy` 文件。

### 3.2 这种做法合理吗

**合理，但比较基础。**

优点：

- 思路清晰；
- 对小数据集更容易训练；
- 相比直接 RGB，光流更适合突出微小运动。

缺点：

- 只使用 onset 和 apex 两帧，时序信息不完整；
- 没做人脸对齐；
- 没做 ROI（眼周、嘴角等局部区域）强化；
- 没做光流标准化或裁剪异常值；
- 预处理失败样本会被训练阶段默默替换成全零张量，不够严谨。

### 3.3 更建议的改进

优先级从高到低：

1. 加入 **人脸检测 + 对齐**；
2. 在预处理阶段剔除失败样本，不要训练时补零；
3. 保存样本清单 CSV，明确哪些样本成功/失败；
4. 加入光流可视化，便于人工核查；
5. 后续尝试 onset-middle-apex 多帧建模；
6. 尝试面部局部区域建模。

## 4. 处理好的数据放在哪里，怎么存

当前默认放在：

- `data/CASME II/processed/u`
- `data/CASME II/processed/v`

命名方式：

- `sub01_EP02_01.npy`
- `sub02_XX.npy`

其中：

- `u/` 存的是水平光流分量；
- `v/` 存的是垂直光流分量。

### 4.1 `.npy` 是什么

`.npy` 是 NumPy 的二进制数组格式。优点：

- 读取快；
- 不像图片那样再次压缩失真；
- 很适合保存模型输入矩阵。

对于本项目来说，`.npy` 里保存的是一个二维矩阵，例如 128×128，表示每个像素点的运动值。

## 5. 水平光流、垂直光流是什么

当两帧图像之间发生运动时，光流会告诉你：

- 某个像素向左/右移动了多少：这就是 **水平光流 `u`**；
- 某个像素向上/下移动了多少：这就是 **垂直光流 `v`**。

因此当前模型其实不是直接看脸，而是在看“脸上的哪个位置往哪边动了多少”。

## 6. 模型是什么

当前主模型是 `SFAMNetLite`：

- 两条分支；
- 一条吃 `u`；
- 一条吃 `v`；
- 每条分支有卷积、池化、CBAM；
- 最后拼接后做分类。

### 6.1 CBAM 是什么

CBAM 是一种轻量注意力模块，分两步：

1. **通道注意力**：判断哪些特征通道更重要；
2. **空间注意力**：判断图像上哪些位置更重要。

可以粗暴理解成：

> 让模型更关注“重要特征”和“重要区域”。

### 6.2 为什么不是 YOLO 那种“基础模型 + 微调”

因为当前任务不是目标检测，而是分类；输入也不是普通 RGB 图像，而是光流矩阵。

YOLO 的思路是：

- 先有一个大规模预训练检测模型；
- 再做目标检测任务微调。

而你这个项目的思路是：

- 直接自己定义一个适合光流输入的小型分类网络；
- 用微表情数据从头训练。

所以这里的“基础模型”不是 ResNet/YOLO 这种外部 backbone，而是项目自己定义的 `SFAMNetLite`。

## 7. 卷积、池化、超参数是怎么定的

### 7.1 当前卷积/池化结构

每条分支的结构大致是：

- Conv(1→32) + BN + ReLU + CBAM + MaxPool
- Conv(32→64) + BN + ReLU + CBAM + MaxPool
- Conv(64→64) + ReLU + MaxPool
- AdaptiveAvgPool 到 4×4
- 拼接后进入全连接层

这是一种比较经典的小型 CNN 设计，核心目的：

- 卷积：提特征；
- 池化：降采样，扩大感受野，减小计算量；
- 全连接：做最终分类。

### 7.2 当前超参数

主训练脚本里现在的关键超参数有：

- batch size = 32
- epochs = 60
- lr = 0.001
- weight decay = 1e-4
- optimizer = AdamW
- scheduler = CosineAnnealingLR
- device = cuda/cpu 自动选择

### 7.3 合理吗

**基本合理，但不一定最优。**

- 60 epoch：对于小数据集可能够，也可能过拟合；
- batch size 32：常规；
- AdamW：合理；
- CosineAnnealing：也合理；
- 但没有早停、没有验证集选模、没有类别重加权，这些都值得补。

## 8. 当前训练是否可视化

原始项目主线几乎没有成体系的可视化。我已补充：

- 每个 fold 的 loss / accuracy 曲线图；
- 每个 fold 的指标 JSON；
- LOSO 总结 JSON。

这些会输出到：

- `outputs/training/`

## 9. 当前训练和评估有哪些硬伤

这是你汇报时一定要知道的部分：

1. **用测试 fold 准确率选“最佳模型”不够规范**。更严谨做法是训练集里再拆验证集。
2. **缺少类别不平衡处理**。微表情任务里这很重要。
3. **准确率不是唯一指标**。应该同时看 UAR / UF1。
4. **预处理失败样本补零** 会污染训练。
5. **没有数据增强**（针对光流输入）或增强策略较弱。
6. **没有独立推理入口**（现在已经补了 `infer_custom.py`）。

## 10. 如何用自己的照片/视频

现在新增了 `infer_custom.py`：

### 10.1 两张图推理

适合你已经有 onset 图和 apex 图时：

```bash
python infer_custom.py --checkpoint checkpoints/best_model_fold_01.pth --image-a path/to/onset.jpg --image-b path/to/apex.jpg
```

### 10.2 视频推理

适合你自己录了一段视频，并且知道想取哪两帧：

```bash
python infer_custom.py --checkpoint checkpoints/best_model_fold_01.pth --video demo.mp4 --onset-frame 3 --apex-frame 12
```

### 10.3 推理输出

脚本会输出：

- 每个类别的概率排序；
- onset/apex 图；
- 光流可视化图；
- `u/v` 光流分量文件。

当前不是实时系统，而是 **离线推理**。

## 11. 可解释性怎么样

目前可解释性一般，不算强。

比较好的地方：

- 可以看光流可视化图，知道模型输入大概是什么；
- 可以看预测概率排序。

不足：

- 没有 Grad-CAM；
- 没有 attention heatmap 导出；
- 没有把 CBAM 的注意力图可视化出来。

后续很适合继续加：

1. 中间特征图保存；
2. CBAM 空间注意力图导出；
3. Grad-CAM。

## 12. 推荐的后续重构方向

为了更规范、模块化、可迭代，建议后续逐步拆分：

- `src/preprocess/`：图像对齐、ROI、光流生成、可视化；
- `src/models/`：backbone、attention、classifier 分开；
- `src/train/`：trainer、evaluator、callbacks；
- `src/infer/`：图片推理、视频推理、可视化导出；
- `configs/`：不同实验配置；
- `reports/`：训练结果、指标、混淆矩阵、图表。

建议优先做的 3 件事：

1. 加一个验证集，不要用测试集选模型；
2. 清理失败样本，不要补零；
3. 补混淆矩阵和 UAR/UF1。
