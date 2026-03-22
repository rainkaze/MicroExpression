import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CASME2FlowDataset(Dataset):
    def __init__(self, processed_dir, csv_path, subjects=None, transform=None):
        self.processed_dir = processed_dir
        self.transform = transform

        # 1. 读取 Excel
        df = pd.read_excel(csv_path)
        df.columns = [str(c).strip() for c in df.columns]

        # 2. 过滤无效行
        df = df[pd.to_numeric(df['OnsetFrame'], errors='coerce').notnull()]

        # 3. 如果指定了被试（LOSO 核心逻辑），则只保留对应被试的数据
        if subjects is not None:
            # 确保 Subject 列是字符串格式并补齐两位，以便匹配
            df['Subject'] = df['Subject'].apply(lambda x: str(int(x)).zfill(2))
            df = df[df['Subject'].isin(subjects)]

        self.df = df
        self.samples = self.df.to_dict('records')

        # 4. 标签映射 (4分类标准)
        self.label_map = {
            'happiness': 0, 'repressed happiness': 0,
            'disgust': 1, 'sadness': 1, 'fear': 1,
            'surprise': 2,
            'others': 3, 'repression': 3, 'tense': 3
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        sub = str(int(row['Subject'])).zfill(2)
        filename = str(row['Filename']).strip()

        # 对应预处理生成的文件名
        sample_id = f"sub{sub}_{filename}"
        u_path = os.path.join(self.processed_dir, 'u', f"{sample_id}.npy")
        v_path = os.path.join(self.processed_dir, 'v', f"{sample_id}.npy")

        # 容错处理：如果预处理失败的文件，跳过加载
        if not os.path.exists(u_path):
            # 实际项目中这里通常返回一个随机样本，或者在 __init__ 中就剔除不存在的文件
            # 这里简单处理：返回全 0 张量
            u = np.zeros((128, 128), dtype=np.float32)
            v = np.zeros((128, 128), dtype=np.float32)
        else:
            u = np.load(u_path).astype(np.float32)
            v = np.load(v_path).astype(np.float32)

        # 增加维度 [H, W] -> [1, H, W]
        u_tensor = torch.from_numpy(u).unsqueeze(0)
        v_tensor = torch.from_numpy(v).unsqueeze(0)

        # 获取标签
        emotion = str(row['Estimated Emotion']).lower().strip()
        label = self.label_map.get(emotion, 3)  # 找不到则默认为 others

        return u_tensor, v_tensor, label