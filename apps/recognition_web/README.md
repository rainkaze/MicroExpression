# 微表情识别展示系统

该目录是 CAS(ME)^3 微表情识别的本地 Web 展示页面，用于毕设系统展示和实验结果说明。页面不参与训练流程。

## 功能

- 自动扫描 `artifacts/runs/*/summary.json` 和各折 `best_model.pt`
- 支持模型切换和多模型对比
- 支持从数据集样本中选择待识别样本
- 支持上传 `.npy` 张量文件
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

上传文件应为 `.npy` 张量：

- `uvd` 模型：形状为 `(3,H,W)`，通道为 `u, v, depth_delta`
- `uv` 模型：形状为 `(2,H,W)` 或从 UVD 自动取前两通道
- `depth` 模型：形状为 `(1,H,W)` 或从 UVD 自动取第三通道

页面中的可视化是模型输入张量的可视化，不是 Grad-CAM 或因果解释。
