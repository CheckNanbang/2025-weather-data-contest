import logging
import os
from datetime import datetime

def get_logger(model_name):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/{model_name}_{now}.log'
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 인코딩 명시
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.log_filename = log_filename
    return logger
