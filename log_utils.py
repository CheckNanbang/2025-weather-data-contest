import os
import csv
import yaml
from datetime import datetime

def save_log(params, metrics, model_name):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # metrics.csv 누적 저장
    metrics_path = "logs/metrics.csv"
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "model", *metrics.keys()])
        if write_header:
            writer.writeheader()
        row = {"timestamp": timestamp, "model": model_name, **metrics}
        writer.writerow(row)
    
    # 상세 로그 저장
    detail_log_path = f"logs/{model_name}_{timestamp}.yaml"
    with open(detail_log_path, "w") as f:
        yaml.dump({"timestamp": timestamp, "model": model_name, "params": params, "metrics": metrics}, f)
