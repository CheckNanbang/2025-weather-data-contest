import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from tqdm import tqdm
import logging
import optuna

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_squared_log_error, mean_absolute_percentage_error, explained_variance_score
)

from dataloader.data_loader import data_loader
from utils.data_split import data_split
from utils.load_params import load_params
from models.train_model import train_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="heat")
    parser.add_argument("--model", type=str, default="XGB")
    parser.add_argument("--split_type", type=str, default="time", choices=["random", "time"])
    parser.add_argument("--model_type", type=str, default="regressor", choices=["regressor", "classifier"])
    parser.add_argument("--submit", action="store_true", help="Submission 생성 여부")
    parser.add_argument("--tune", action="store_true", help="Optuna로 하이퍼파라미터 튜닝")
    parser.add_argument("--params", nargs="*", help="key=value 형태로 하이퍼파라미터 입력")
    
    return parser.parse_args()


def parse_params(param_list):
    if not param_list:
        return {}
    return {k: eval(v) for k, v in (p.split('=') for p in param_list)}


def setup_logger(log_name):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_name)
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
    return logging.getLogger()


def evaluate(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "EVS": explained_variance_score(y_true, y_pred)
    }
    try:
        metrics["MSLE"] = mean_squared_log_error(y_true, y_pred)
    except ValueError:
        metrics["MSLE"] = None
    return metrics


def objective(trial, model_name, model_type, x_train, y_train, x_valid, y_valid):
    if model_name == "XGB":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10)
        }
    else:
        raise NotImplementedError("Only XGB tuning implemented")

    model = train_model(model_name, model_type, params, x_train, y_train)
    preds = model.predict(x_valid)
    return mean_squared_error(y_valid, preds)


def main():
    args = parse_args()
    now = datetime.now()
    date_code = now.strftime("%m%d%H%M")
    save_name = f"{args.model}_{args.dataset}_{args.split_type}_{date_code}.csv"
    logger = setup_logger(f"{args.model}_{date_code}.log")

    # 데이터 로딩 및 분할
    train_df, test_df, submission_df, target_column = data_loader(args.dataset)
    x_train, x_valid, y_train, y_valid = data_split(args.split_type, train_df, target_column)
    X_test = test_df.copy()

    # 컬럼 삭제
    drop_cols = ["train_heatbranch_id"]
    for col in drop_cols:
        x_train = x_train.drop(columns=[col], errors="ignore")
        x_valid = x_valid.drop(columns=[col], errors="ignore")
        X_test = X_test.drop(columns=[col], errors="ignore")

    # 파라미터 준비
    user_params = parse_params(args.params)
    default_params = load_params(args.model, args.model_type)
    params = {**default_params, **user_params}
    
    print("🔧 최종 사용 파라미터:")
    for k, v in params.items():
        print(f"{k}: {v}")
        logger.info(f"{k}: {v}")

    # Optuna 하이퍼파라미터 튜닝
    if args.tune:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args.model, args.model_type, x_train, y_train, x_valid, y_valid), n_trials=20)
        print("🎯 Best trial:", study.best_trial.params)
        params.update(study.best_trial.params)

    print("🚀 모델 학습 시작...")
    model = train_model(args.model, args.model_type, params, x_train, y_train)
    print("✅ 학습 완료!")

    # 예측 및 평가
    y_valid_pred = model.predict(x_valid)
    metrics = evaluate(y_valid, y_valid_pred)

    print("📊 검증 지표")
    for k, v in metrics.items():
        result = f"{k}: {v:.4f}" if v is not None else f"{k}: Not available"
        print(result)
        logger.info(result)

    # 최종 학습 및 예측
    if args.submit:
        print("📝 최종 모델 학습 및 submission 생성")
        x_total = pd.concat([x_train, x_valid], axis=0)
        y_total = pd.concat([y_train, y_valid], axis=0)
        model = train_model(args.model, args.model_type, params, x_total, y_total)
        X_test = X_test.drop(columns=[target_column], errors="ignore")
        test_pred = model.predict(X_test)
        submission_df['heat_demand'] = test_pred
        submission_df.to_csv(save_name, index=False, encoding='utf-8-sig')
        print(f"📁 제출 파일 저장 완료: {save_name}")
        logger.info(f"Submission saved to {save_name}")


if __name__ == "__main__":
    main()
