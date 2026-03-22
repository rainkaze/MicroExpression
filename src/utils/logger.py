import logging
import os


def setup_logger(log_dir, fold_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(fold_name)
    logger.setLevel(logging.INFO)

    # 防止日志重复输出
    if not logger.handlers:
        # 文件处理器
        fh = logging.FileHandler(os.path.join(log_dir, f"{fold_name}.log"))
        fh.setLevel(logging.INFO)

        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式设置
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger