import numpy as np
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn as sns
from models.train_model import train_model

def objective(trial, model_name, x_train, y_train, x_valid, y_valid):
    """
    Optuna를 위한 목적 함수. 하이퍼파라미터를 튜닝함.
    
    Args:
        trial (optuna.Trial): Optuna 트라이얼 객체
        model_name (str): 튜닝할 모델 이름
        x_train (pd.DataFrame): 학습 입력 데이터
        y_train (pd.Series): 학습 타겟 데이터
        x_valid (pd.DataFrame): 검증 입력 데이터
        y_valid (pd.Series): 검증 타겟 데이터
        
    Returns:
        float: 최소화할 목적 함수 값 (RMSE)
    """
    if model_name == "XGB":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "random_state": 42
        }
    elif model_name == "LGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42
        }
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")

    # 모델 학습
    model = train_model(model_name, params, x_train, y_train)
    
    # 예측 및 평가
    preds = model.predict(x_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    
    return rmse


def plot_results(model, x_valid, y_valid, y_pred, save_path="plots"):
    """
    모델 예측 결과를 시각화하는 함수.
    
    Args:
        model: 학습된 모델 객체
        x_valid (pd.DataFrame): 검증 입력 데이터
        y_valid (pd.Series): 실제 타겟 값
        y_pred (np.ndarray): 모델이 예측한 값
        save_path (str): 플롯을 저장할 경로
    """
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 실제값 vs 예측값 산점도
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_valid, y=y_pred, alpha=0.5)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], 'r--')
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.title('true vs predicted values')
    plt.savefig(os.path.join(save_path, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 잔차 히스토그램
    residuals = y_valid - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('residuals')
    plt.ylabel('frequency')
    plt.title('Residuals Histogram')
    plt.axvline(0, color='r', linestyle='--')
    plt.savefig(os.path.join(save_path, 'residuals_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 특성 중요도 (지원되는 경우)
    try:
        importances = model.feature_importance()
        top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20])
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(top_features.values()), y=list(top_features.keys()))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("특성 중요도를 시각화할 수 없습니다.")
    
    print(f"시각화 결과가 {save_path} 디렉토리에 저장되었습니다.")