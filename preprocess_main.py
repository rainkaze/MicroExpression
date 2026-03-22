import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from src.preprocess.flow_utils import OpticalFlowEngine


class CASME2Preprocessor:
    def __init__(self, root_dir, csv_path, output_dir, target_size=(128, 128)):
        self.root_dir = os.path.abspath(root_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.target_size = target_size

        try:
            self.df = pd.read_excel(csv_path)
            self.df.columns = [str(c).strip() for c in self.df.columns]
            # 过滤非数字行
            self.df = self.df[pd.to_numeric(self.df['OnsetFrame'], errors='coerce').notnull()]
            print(f"✅ Excel 加载成功，待处理样本: {len(self.df)}")
        except Exception as e:
            print(f"❌ 读取 Excel 失败: {e}")

        self.flow_engine = OpticalFlowEngine(algorithm='TV-L1')
        os.makedirs(os.path.join(output_dir, 'u'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'v'), exist_ok=True)

    def _get_actual_path(self, sub_id, filename):
        """
        深度搜索策略：在 root_dir 下寻找包含 filename 的文件夹
        解决有的目录是 sub01，有的是 01，或者多了一层目录的问题
        """
        # 尝试 1: 标准路径 root/sub01/EP02_01
        path1 = os.path.join(self.root_dir, f"sub{sub_id}", filename)
        if os.path.exists(path1): return path1

        # 尝试 2: 简写路径 root/01/EP02_01
        path2 = os.path.join(self.root_dir, sub_id, filename)
        if os.path.exists(path2): return path2

        # 尝试 3: 模糊搜索 (解决文件夹名大小写或微小差异)
        for root, dirs, files in os.walk(self.root_dir):
            if filename in dirs:
                return os.path.join(root, filename)
        return None

    def _load_img(self, case_path, frame_idx):
        """
        尝试所有可能的图片命名格式
        """
        if not case_path: return None
        # CASME II 常见的命名方案
        patterns = [f"img{frame_idx}.jpg", f"img{str(frame_idx).zfill(3)}.jpg", f"img{str(frame_idx).zfill(5)}.jpg"]

        for p in patterns:
            full_path = os.path.join(case_path, p)
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None: return img
        return None

    def run(self):
        success_count = 0
        print(f"🔍 正在扫描数据源: {self.root_dir}")

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                sub = str(int(row['Subject'])).zfill(2)
                filename = str(row['Filename']).strip()
                onset = int(row['OnsetFrame'])
                apex = int(row['ApexFrame'])

                # 获取真实目录
                case_path = self._get_actual_path(sub, filename)

                # 加载图片
                img_onset = self._load_img(case_path, onset)
                img_apex = self._load_img(case_path, apex)

                if img_onset is None or img_apex is None:
                    continue  # 找不到图，跳过

                # 缩放与光流计算
                img_onset = cv2.resize(img_onset, self.target_size)
                img_apex = cv2.resize(img_apex, self.target_size)

                u, v = self.flow_engine.compute_flow(img_onset, img_apex)

                # 保存
                sample_id = f"sub{sub}_{filename}"
                np.save(os.path.join(self.output_dir, 'u', f"{sample_id}.npy"), u.astype(np.float16))
                np.save(os.path.join(self.output_dir, 'v', f"{sample_id}.npy"), v.astype(np.float16))
                success_count += 1

            except Exception:
                continue

        print(f"\n✨ 处理完毕！")
        print(f"📊 成功: {success_count} | 失败: {len(self.df) - success_count}")
        if success_count == 0:
            print("❌ 依然没找到图片。请手动确认以下文件是否存在：")
            print(
                f"   D:/Projects/PyCharmProjects/MicroExpression/data/CASME II/CASME2_RAW/sub01/EP02_01f/ 下是否有 .jpg 文件？")


if __name__ == "__main__":
    processor = CASME2Preprocessor(
        root_dir='./data/CASME II/CASME2_RAW',
        csv_path='./data/CASME II/CASME2-coding-20140508.xlsx',
        output_dir='./data/CASME II/processed'
    )
    processor.run()