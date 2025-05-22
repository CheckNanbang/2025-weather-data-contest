import os
import logging
import csv

def setup_logger(log_name):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_name)
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, encoding='utf-8-sig')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def save_metrics_to_csv(date_code, model_name, metrics, csv_path="result.csv"):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date_code", "model", "MAE", "MSE", "RMSE", "R2", "EVS", "MAPE", "MSLE"])
        for metric in metrics:
            row = [
                date_code,
                model_name,
                metric.get("MAE"),
                metric.get("MSE"),
                metric.get("RMSE"),
                metric.get("R2"),
                metric.get("EVS"),
                metric.get("MAPE"),
                metric.get("MSLE")
            ]
            writer.writerow(row)
