#] utils/evaluation.py
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

def evaluate(y_true, y_pred):
    """
    회귀 모델의 성능을 평가하는 여러 지표를 계산.
    
    Args:
        y_true (array-like): 실제 타겟 값
        y_pred (array-like): 모델이 예측한 값
        
    Returns:
        dict: 계산된 모든 성능 지표를 포함하는 딕셔너리
    """
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "EVS": explained_variance_score(y_true, y_pred)
    }
    
    # 음수 값이 있는 경우 MAPE가 발산할 수 있으므로 처리
    try:
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics["MAPE"] = None
        print("경고: MAPE를 계산할 수 없습니다 (음수 값 또는 0 값이 존재함)")
    
    # MSLE는 음수 값이 있을 경우 계산할 수 없음
    if np.all(y_true > 0) and np.all(y_pred > 0):
        try:
            from sklearn.metrics import mean_squared_log_error
            metrics["MSLE"] = mean_squared_log_error(y_true, y_pred)
        except:
            metrics["MSLE"] = None
    else:
        metrics["MSLE"] = None
    
    return metrics