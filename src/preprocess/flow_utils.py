import cv2


class OpticalFlowEngine:
    def __init__(self, algorithm="TV-L1"):
        self.algorithm = algorithm
        self.engine = None

        if algorithm == "TV-L1":
            try:
                self.engine = cv2.optflow.createOptFlow_DualTVL1()
                print("Loaded TV-L1 optical flow engine")
            except AttributeError:
                print("Warning: cv2.optflow is unavailable; falling back to Farneback.")
                self.algorithm = "Farneback"

    def compute_flow(self, prev_frame, next_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if self.algorithm == "TV-L1" and self.engine is not None:
            flow = self.engine.calc(prev_gray, next_gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )

        return flow[..., 0], flow[..., 1]
