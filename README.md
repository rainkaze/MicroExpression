# MicroExpression

本仓库用于完成 `CAS(ME)^3 Part A ME clip` 上的微表情识别实验与本地展示系统。当前任务边界是 **recognition-only**：只做 clip-level 微表情类别识别，不做 spotting、滑窗检测或伪标签定位。

论文方向：

```text
基于注意网络的微表情识别
```

当前工程主线：

```text
基于 u / v / depth_delta 的场景流注意融合微表情识别
```

## 环境

安装依赖：

```powershell
pip install -r requirements.txt
```

主要依赖包括 PyTorch、OpenCV、NumPy、Pandas、Scikit-learn、Matplotlib、Pillow 和 OpenPyXL。

## 数据与输入定义

当前处理后的数据根目录：

```text
data/processed/casme3_scene_flow/
```

核心 manifest：

```text
data/processed/casme3_scene_flow/casme3_scene_flow_manifest.csv
```

模型输入：

```text
uv:    2 通道，(u, v)
depth: 1 通道，(depth_delta)
uvd:   3 通道，(u, v, depth_delta)
```

其中：

- `u`：起始帧到峰值帧之间的水平光流。
- `v`：起始帧到峰值帧之间的垂直光流。
- `depth_delta`：峰值帧深度减去起始帧深度，即 `apex_depth - onset_depth`。它不是单帧原始深度图。


## 数据处理

生成 CAS(ME)^3 识别 manifest：

```powershell
python scripts/data/build_casme3_manifest.py
```

默认读取：

```text
data/raw/CAS(ME)^3/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx
data/raw/CAS(ME)^3/Part_A_ME_clip/
```

生成场景流张量：

```powershell
python scripts/preprocess/build_casme3_scene_flow.py
```

输出：

```text
data/processed/casme3_scene_flow/uv/*.npy
data/processed/casme3_scene_flow/depth/*.npy
data/processed/casme3_scene_flow/uvd/*.npy
data/processed/casme3_scene_flow/build_report.md
```

当前数据状态：

```text
标注行数: 860
识别 clean 样本: 857
subject 数: 94
预处理失败: 0
```

## 模型

当前有效模型名称：

```text
uv_baseline
depth_baseline
uvd_concat
uvd_attention
uvd_masked_attention
uvd_residual_masked_attention
```

含义：

- `uv_baseline`：只使用二维光流 `u, v`。
- `depth_baseline`：只使用 `depth_delta`。
- `uvd_concat`：直接拼接 `u, v, depth_delta`。
- `uvd_attention`：三分支注意/门控融合模型，是当前四分类主线。
- `uvd_masked_attention`：显式空间 mask 抑制版本，用于诊断消融。
- `uvd_residual_masked_attention`：残差式 mask 增强版本，用于诊断消融和七分类对比。

模型入口：

```text
src/models/scene_flow_models.py
src/models/masked_scene_flow.py
```

## 训练与评估

训练统一入口：

```powershell
python scripts/train/train_recognition.py --config path/to/config.toml
```

也可以使用 `scripts/train/` 下的一键脚本，例如：

```powershell
python scripts/train/main/run_uvd4_attention_5fold.py
python scripts/train/main/run_uvd7_attention_5fold.py
```

评估协议：

```text
subject-aware stratified 5-fold cross-validation
```

该协议使用 `StratifiedGroupKFold`，以 subject 为 group，避免同一被试同时出现在训练集和测试集，同时尽量保持各 fold 类别分布接近。

优先报告指标：

```text
macro_f1
uar
accuracy
class-wise recall / F1
confusion matrix
```

CAS(ME)^3 类别不均衡明显，不应只用 accuracy 判断模型优劣。

## 当前实验

截至当前阶段，已完成 14 组 5-fold 实验：

```text
uv4_baseline_5fold
depth4_baseline_5fold
uvd4_concat_5fold
uvd4_attention_5fold
uvd4_attention_focal_sampler_5fold
uvd4_masked_attention_5fold
uvd4_residual_masked_attention_5fold

uv7_baseline_5fold
depth7_baseline_5fold
uvd7_concat_5fold
uvd7_attention_5fold
uvd7_attention_focal_sampler_5fold
uvd7_masked_attention_5fold
uvd7_residual_masked_attention_5fold
```

结果汇总脚本：

```powershell
python scripts/analysis/summarize_scene_flow_runs.py
```

汇总输出：

```text
artifacts/analysis/scene_flow_run_summary.csv
artifacts/analysis/scene_flow_run_class_metrics.csv
artifacts/analysis/scene_flow_run_confusion_matrices.json
artifacts/analysis/scene_flow_run_summary.md
```

如果 CSV 被 Excel、WPS 或 PyCharm 预览占用，脚本会自动写入带时间戳的新文件。

当前主要结论：

- 四分类中，`uvd_attention` 的 Macro-F1 和 UAR 最好，对 `positive` 类召回改善明显。
- 七分类中，`uv_baseline` 的 Macro-F1 较稳，`uvd_residual_masked_attention` 的 Accuracy 和 UAR 较高，但小类仍然困难。
- `depth_baseline` 单独较弱，说明 depth_delta 不能独立承担识别任务。
- `uvd_concat` 不稳定，说明简单拼接 depth 不等于有效融合。
- focal sampler 未稳定提升当前 UVD attention，适合作为训练策略消融的负向结果。

## 本地展示系统

启动：

```powershell
python scripts/app/run_recognition_web.py --host 127.0.0.1 --port 7860 --device cpu
```

访问：

```text
http://127.0.0.1:7860
```

功能：

- 模型切换。
- 多模型预测对比。
- 数据集样本选择。
- 上传 onset/apex RGB 图片对进行预测。
- 可选上传 onset/apex depth 图以支持 UVD/depth 模型。
- 上传 `.npy` 张量用于研究复现实验。
- 展示原始图像、运动/深度输入可视化、类别概率和通道统计。
- 支持 CPU/GPU 推理设备切换。

说明：

- 普通用户更适合使用“上传图片”入口。
- `.npy` 是预处理后的模型输入张量，主要用于研究复现和调试。
- 页面中的 `u/v/depth_delta/flow_magnitude/depth_abs` 是输入可视化，不是 Grad-CAM 或因果解释。
- 数据集样本模式用于系统展示和模型行为分析，不应替代正式 5-fold 测试结果。

## 目录结构

```text
apps/recognition_web/              # 本地展示系统
configs/train/casme3/              # 训练配置
data/processed/casme3_scene_flow/  # 当前处理后的识别数据
scripts/analysis/                  # 结果汇总与分析
scripts/app/                       # Web 启动脚本
scripts/data/                      # manifest 构建
scripts/preprocess/                # 预处理
scripts/train/                     # 训练入口
src/datasets/                      # 数据集读取
src/models/                        # 模型结构
src/preprocess/                    # 光流和 depth_delta 构建函数
src/training/                      # 训练、损失、指标、报告
```

## 许可证

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。