# utils/log_utils.py
import os
import logging
import csv

def setup_logger(log_name):
    """
    로깅 설정을 초기화하는 함수.
    
    Args:
        log_name (str): 로그 파일 이름
        
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_name)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # 모든 로그 레벨 저장

    # 파일 핸들러 (한글 깨짐 방지)
    file_handler = logging.FileHandler(log_path, encoding='utf-8-sig')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    # 기존 핸들러 제거 후 새 핸들러 추가
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_metrics_to_csv(date_code, model_name, metrics, csv_path="result.csv"):
    row = [
        date_code,
        model_name,
        metrics.get("MAE"),
        metrics.get("MSE"),
        metrics.get("RMSE"),
        metrics.get("R2"),
        metrics.get("EVS"),
        metrics.get("MAPE"),
        metrics.get("MSLE")
    ]
    # 파일이 없으면 헤더 추가
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date_code", "model", "MAE", "MSE", "RMSE", "R2", "EVS", "MAPE", "MSLE"])
        writer.writerow(row)

