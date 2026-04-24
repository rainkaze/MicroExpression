# MicroExpression

基于注意网络的微表情识别

项目当前目标：

- 基于 `CAS(ME)^3 Part A ME clip` 做微表情 `clip-level recognition`
- 从原始数据重新生成 `manifest` 和运动表征
- 训练 `4 分类` 与 `7 分类` 模型
- 自动输出指标、曲线图、混淆矩阵和简报

**当前任务范围**

- 任务类型：微表情识别
- 数据集：`CAS(ME)^3 Part A ME clip`
- 标签体系：
  - `7分类`：`disgust / surprise / others / fear / anger / sad / happy`
  - `4分类`：`negative / positive / surprise / others`
- 输入形式：
  - `flow`：4 通道运动张量，包含 `u / v / magnitude / orientation`
  - `rgbd_flow`：5 通道张量，包含 `flow + depth_delta`
- 划分方式：训练引擎中使用 `subject-aware` fold 划分

**目录结构**

```text
MicroExpression/
├─ configs/
│  └─ train/casme3/
├─ data/
│  ├─ raw/CAS(ME)^3/
│  └─ processed/casme3_recognition_v2/
├─ scripts/
│  ├─ data/
│  ├─ preprocess/
│  ├─ train/
│  └─ visualize/
├─ src/
│  ├─ datasets/
│  ├─ models/
│  ├─ preprocess/
│  ├─ training/
│  └─ utils/
├─ artifacts/
│  └─ runs_dev2/
├─ SFAMNet-main/
├─ requirements.txt
└─ README.md
```

**数据流程**

当前有效流程是：

1. 从原始标注表和 clip 目录重新生成清洗后的 `manifest`
2. 校验 onset/apex/offset 对应的 RGB 和 depth 帧
3. 从原始帧重新生成 `flow` 和 `rgbd_flow`
4. 基于新生成的 `manifest` 训练识别模型
5. 输出结果、图表和报告

原始数据默认位置：

- 标注文件：
  - `data/raw/CAS(ME)^3/annotation/cas(me)3_part_A_ME_label_JpgIndex_v2.xlsx`
- clip 根目录：
  - `data/raw/CAS(ME)^3/Part_A_ME_clip`

处理后输出目录：

- `data/processed/casme3_recognition_v2`

**当前数据状态**

根据最新重建结果：

- 原始标注总行数：`860`
- 清洗后可用样本数：`854`
- 受试者数：`94`
- 成功生成运动张量样本数：`854`
- 预处理失败数：`0`

`7分类` 分布：

- `anger`: `64`
- `disgust`: `249`
- `fear`: `85`
- `happy`: `55`
- `others`: `157`
- `sad`: `57`
- `surprise`: `187`

`4分类` 分布：

- `negative`: `455`
- `others`: `157`
- `positive`: `55`
- `surprise`: `187`


**环境配置**

建议环境：

```powershell
conda create -n micro_expression python=3.11 -y
conda activate micro_expression
pip install -r requirements.txt
```

依赖文件：

- [requirements.txt](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/requirements.txt)

**第一步：生成清洗后的 Manifest**

运行：

```powershell
python scripts/data/build_casme3_manifest.py
```

输出：

- [casme3_manifest.csv](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/data/processed/casme3_recognition_v2/casme3_manifest.csv)
- [casme3_manifest_audit.md](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/data/processed/casme3_recognition_v2/casme3_manifest_audit.md)

可选参数：

```powershell
python scripts/data/build_casme3_manifest.py --annotation <xlsx路径> --clip-root <clip根目录> --output-dir <输出目录>
```

**第二步：生成运动张量**

运行：

```powershell
python scripts/preprocess/build_casme3_motion.py
```

输出：

- [casme3_recognition_manifest.csv](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/data/processed/casme3_recognition_v2/casme3_recognition_manifest.csv)
- `data/processed/casme3_recognition_v2/flow/*.npy`
- `data/processed/casme3_recognition_v2/rgbd_flow/*.npy`
- [build_report.md](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/data/processed/casme3_recognition_v2/build_report.md)

可选参数：

```powershell
python scripts/preprocess/build_casme3_motion.py --image-size 128 --algorithm TV-L1
```

支持的光流算法：

- `TV-L1`
- `Farneback`

**第三步：训练模型**

通用命令行入口：

```powershell
python scripts/train/train_recognition.py --config configs/train/casme3/flow_multibranch_7class_v2_sampler07.toml
```

PyCharm 一键入口：

- [run_flow_multibranch_4class.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_flow_multibranch_4class.py)
- [run_flow_multibranch_7class.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_flow_multibranch_7class.py)
- [run_flow_multibranch_7class_v2.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_flow_multibranch_7class_v2.py)
- [run_flow_multibranch_7class_v2_sampler07.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_flow_multibranch_7class_v2_sampler07.py)
- [run_flow_multibranch_7class_v2_weighted_ce.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_flow_multibranch_7class_v2_weighted_ce.py)
- [run_rgbd_fusion_7class.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/scripts/train/run_rgbd_fusion_7class.py)

**训练时你会看到什么**

现在训练不会静默运行。你自己启动训练时，控制台会显示：

- 每个 epoch 开始提示
- 训练集 batch 进度条
- 验证集 batch 进度条
- 每个 epoch 结束后的核心指标：
  - `train_loss`
  - `train_f1`
  - `val_loss`
  - `val_f1`
  - `val_acc`
  - `val_uar`

如果控制台还在刷新 `epoch` 或 batch 进度条，就不是卡死。

**当前配置文件**

位于 `configs/train/casme3/`：

- `flow_multibranch_4class.toml`
  - `flow` 4分类基础版
- `flow_multibranch_7class.toml`
  - `flow` 7分类基础版
- `rgbd_fusion_7class.toml`
  - `rgbd_flow` 7分类基础版
- `flow_multibranch_7class_v2.toml`
  - 更强的 `flow` 7分类版本
  - 使用 `balanced sampler + focal`
- `flow_multibranch_7class_v2_weighted_ce.toml`
  - `v2` 骨架
  - 使用 `balanced sampler + weighted_ce`
- `flow_multibranch_7class_v2_sampler07.toml`
  - 当前最优配置
  - 使用 `balanced sampler(power=0.7) + focal`

**训练输出内容**

每次训练会写入：

- `artifacts/runs_dev2/<run_name>/`

每个 fold 目录包含：

- `best_model.pt`
- `result.json`
- `history.csv`
- `training_curves.png`
- `confusion_matrix.png`
- `report.md`

run 根目录包含：

- `summary.json`
- `summary.md`

示例：

```text
artifacts/runs_dev2/flow_multibranch_7class_v2_sampler07/
├─ fold_0/
│  ├─ best_model.pt
│  ├─ result.json
│  ├─ history.csv
│  ├─ training_curves.png
│  ├─ confusion_matrix.png
│  └─ report.md
├─ summary.json
└─ summary.md
```

**结果可视化**

如果训练已经完成，想重新生成图表：

```powershell
python scripts/visualize/plot_training_run.py --run-dir artifacts/runs_dev2/flow_multibranch_7class_v2_sampler07
```

会重新生成：

- 训练曲线图
- 混淆矩阵图
- fold 报告
- run 汇总报告

**当前有效模型**

模型实现文件：

- [motion_baselines.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/models/motion_baselines.py)
- [common.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/models/common.py)

当前活跃模型族：

- `apex_baseline`
  - RGB apex 图像分类
- `motion_multibranch`
  - 轻量三分支 `flow` 基线
- `motion_multibranch_v2`
  - 四分支 `flow` 主线模型
  - 分别处理 `u / v / mag / ori`
- `rgbd_fusion`
  - `RGB-D motion` 融合基线

**训练引擎**

核心文件：

- [engine.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/training/engine.py)
- [losses.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/training/losses.py)
- [metrics.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/training/metrics.py)
- [splits.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/training/splits.py)
- [reporting.py](/abs/path/D:/Projects/PyCharmProjects/MicroExpression/src/training/reporting.py)

当前已实现：

- `subject-aware` fold 划分
- `weighted cross-entropy`
- `focal loss`
- `balanced sampler`
- 基于验证集 `macro_f1` 的 early stopping
- 控制台 epoch 日志
- batch 级进度条
- 自动导出图表和报告

**当前实验结果**

当前 `fold_0` 结果如下：

- `flow_multibranch_4class`
  - `Acc 0.4790`
  - `Macro-F1 0.3677`
  - `UAR 0.3802`
- `flow_multibranch_7class`
  - `Acc 0.2294`
  - `Macro-F1 0.1896`
  - `UAR 0.1997`
- `rgbd_fusion_7class`
  - `Acc 0.2176`
  - `Macro-F1 0.1430`
  - `UAR 0.1567`
- `flow_multibranch_7class_v2`
  - `Acc 0.3353`
  - `Macro-F1 0.2560`
  - `UAR 0.2521`
- `flow_multibranch_7class_v2_weighted_ce`
  - `Acc 0.3706`
  - `Macro-F1 0.2490`
  - `UAR 0.2611`
- `flow_multibranch_7class_v2_sampler07`
  - `Acc 0.3941`
  - `Macro-F1 0.2631`
  - `UAR 0.2723`

这些都还是开发阶段的单 fold 结果，不是最终论文结果。正式汇报前应对同一配置做完整多 fold 评估。


**许可证**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。
