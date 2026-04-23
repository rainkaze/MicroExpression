import logging
import os
from datetime import datetime


def setup_logger(log_dir, fold_name):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(fold_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_path = os.path.join(log_dir, f"{fold_name}.log")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    try:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(log_dir, f"{fold_name}_{stamp}.log")
        try:
            file_handler = logging.FileHandler(fallback_path, encoding="utf-8")
        except PermissionError:
            file_handler = None

    if file_handler is not None:
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
