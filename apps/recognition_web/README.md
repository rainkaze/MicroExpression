# 微表情识别展示系统

该目录是 CAS(ME)^3 微表情识别的本地 Web 展示页面，用于毕设系统展示和实验结果说明。页面不参与训练流程。

## 功能

- 自动扫描 `artifacts/runs/*/summary.json` 和各折 `best_model.pt`
- 支持模型切换和多模型对比
- 支持从数据集样本中选择待识别样本
- 支持上传 onset/apex RGB 图片对进行预测
- 支持上传 `.npy` 张量文件，供研究复现实验使用
- 支持 CPU/GPU 推理设备切换
- 展示预测类别、置信度和各类别概率
- 展示输入通道可视化：`u`、`v`、`depth_delta`、光流幅值和深度幅值
- 展示通道统计量，辅助说明运动强弱和深度变化

## 启动

```powershell
D:\CodeTools\miniconda\envs\micro_expression\python.exe D:\Projects\PyCharmProjects\MicroExpression\scripts\app\run_recognition_web.py --host 127.0.0.1 --port 7860 --device cpu
```

浏览器访问：

```text
http://127.0.0.1:7860
```

如需使用 GPU，可将 `--device cpu` 改为 `--device cuda`。

## 输入说明

### 图片上传

普通用户更适合使用图片上传入口：

- 必选：`onset RGB` 和 `apex RGB`
- 可选：`onset depth` 和 `apex depth`

只上传 RGB 起始帧和峰值帧时，系统会计算光流，只能运行 `uv` 模型。若要运行 `uvd` 或 `depth` 模型，需要同时上传两张 depth 图。

### 张量上传

上传文件应为 `.npy` 张量：

- `uvd` 模型：形状为 `(3,H,W)`，通道为 `u, v, depth_delta`
- `uv` 模型：形状为 `(2,H,W)` 或从 UVD 自动取前两通道
- `depth` 模型：形状为 `(1,H,W)` 或从 UVD 自动取第三通道

`.npy` 是预处理后的模型输入，主要面向实验复现和内部调试；普通用户通常不会手工准备该文件。

页面中的运动与深度可视化是模型输入张量的可视化，不是 Grad-CAM 或因果解释。
