import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

def evaluate(y_true, y_pred):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "EVS": explained_variance_score(y_true, y_pred)
    }
    try:
        metrics["MAPE"] = mean_absolute_percentage_error(y_true, y_pred)
    except:
        metrics["MAPE"] = None
        print("경고: MAPE를 계산할 수 없습니다 (음수 값 또는 0 값이 존재함)")
    if np.all(y_true > 0) and np.all(y_pred > 0):
        try:
            from sklearn.metrics import mean_squared_log_error
            metrics["MSLE"] = mean_squared_log_error(y_true, y_pred)
        except:
            metrics["MSLE"] = None
    else:
        metrics["MSLE"] = None
    return metrics
