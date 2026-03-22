import cv2
import numpy as np


class OpticalFlowEngine:
    def __init__(self, algorithm='TV-L1'):
        self.algorithm = algorithm
        self.engine = None

        if algorithm == 'TV-L1':
            try:
                # 尝试初始化 Dual TV-L1
                self.engine = cv2.optflow.createOptFlow_DualTVL1()
                print("✅ 成功加载 TV-L1 光流引擎")
            except AttributeError:
                print("⚠️ 警告: 未发现 cv2.optflow 模块。请运行 'pip install opencv-contrib-python'")
                print("🔄 自动切换至备选方案: Farneback 算法")
                self.algorithm = 'Farneback'

    def compute_flow(self, prev_frame, next_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if self.algorithm == 'TV-L1' and self.engine is not None:
            flow = self.engine.calc(prev_gray, next_gray, None)
        else:
            # 稳健性备选方案：Farneback
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, next_gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

        u = flow[..., 0]
        v = flow[..., 1]
        return u, v