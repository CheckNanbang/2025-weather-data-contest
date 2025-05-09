import argparse
import yaml
import pandas as pd
from tqdm import trange
from datetime import datetime
import os
import json

from logger import get_logger
from data.data_loader import XGBoostDataLoader, RFDataLoader
from models.xgboost_model import XGBoostModel
from models.rf_model import RFModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_params(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def save_params(filename, params):
    with open(filename, 'w') as f:
        yaml.dump(params, f)

def tune_hyperparams(params, key, value, filename):
    params[key] = value
    save_params(filename, params)

def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def save_metrics_csv(model_name, metrics_dict):
    if not os.path.exists('results'):
        os.makedirs('results')
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'results/metrics_{model_name}_{now}.csv'
    df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, index=False)
    return csv_path

def save_metrics_cumulative(model_name, metrics):
    csv_path = 'results/metrics_cumulative.csv'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data = pd.DataFrame([{
        'Model': model_name,
        'DateTime': now,
        'MAE': metrics['MAE'],
        'MSE': metrics['MSE'],
        'RMSE': metrics['RMSE'],
        'R2': metrics['R2']
    }])
    if not os.path.exists('results'):
        os.makedirs('results')
    if os.path.exists(csv_path):
        new_data.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    else:
        new_data.to_csv(csv_path, mode='w', header=True, index=False, encoding='utf-8')

def train_and_evaluate(model_name, n_epochs):
    logger = get_logger(model_name)

    if model_name == 'xgboost':
        data_loader = XGBoostDataLoader('data/')
        params = load_params('params/xgboost_params.yaml')
        model = XGBoostModel(params)
    elif model_name == 'rf':
        data_loader = RFDataLoader('data/')
        params = load_params('params/rf_params.yaml')
        model = RFModel(params)
    else:
        raise ValueError('지원하지 않는 모델입니다.')

    # 모델 파라미터를 로그에 남김 (json 포맷)
    logger.info(f'모델 파라미터: {json.dumps(params, ensure_ascii=False)}')

    X_train, X_test, y_train, y_test = data_loader.load_data()
    best_score = float('-inf')
    best_metrics = {}

    for epoch in trange(1, n_epochs + 1, desc=f"Training {model_name}", unit="epoch"):
        model.train(X_train, y_train)
        y_pred = model.model.predict(X_test)
        metrics = get_metrics(y_test, y_pred)
        logger.info(f'Epoch {epoch}: ' + ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()]))
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_metrics = metrics

    logger.info(f'최종 평가지표 (Best R2 기준): ' + ', '.join([f'{k}={v:.4f}' for k, v in best_metrics.items()]))
    csv_path = save_metrics_csv(model_name, best_metrics)
    logger.info(f'최종 평가지표가 {csv_path}에 저장되었습니다.')
    save_metrics_cumulative(model_name, best_metrics)
    logger.info(f'누적 평가지표가 results/metrics_cumulative.csv에 저장되었습니다.')
    logger.info(f'로그 파일: {logger.log_filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='모델 및 에포크 선택 실행')
    parser.add_argument('--model', type=str, required=True, choices=['xgboost', 'rf'], help='실행할 모델 이름')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에포크 수 (기본값: 5)')
    args = parser.parse_args()
    train_and_evaluate(args.model, args.epochs)
