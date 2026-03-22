import os


class ProjectConfig:
    # 基础目录定义
    BASE_DATA_PATH = './data/CASME II'

    # 输入路径
    RAW_DATA_DIR = os.path.join(BASE_DATA_PATH, 'CASME2_RAW')
    METADATA_CSV = os.path.join(BASE_DATA_PATH, 'CASME2-coding-20140508.xlsx')

    # 输出路径
    PROCESSED_DIR = os.path.join(BASE_DATA_PATH, 'processed')
    LOG_DIR = './logs'
    CHECKPOINT_DIR = './checkpoints'

    # 训练超参数
    BATCH_SIZE = 32
    EPOCHS = 60
    LEARNING_RATE = 0.001
    NUM_CLASSES = 4