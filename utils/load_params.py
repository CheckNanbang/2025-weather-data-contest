import os
import yaml

def load_params(model_name):
    supported_models = ["LGBM", "XGB"]
    params_path = os.path.join("models", "params")
    if model_name not in supported_models:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}. 지원되는 모델: {supported_models}")
    param_files = {
        "LGBM": "lgbm_param.yaml",
        "XGB": "xgb_param.yaml"
    }
    param_file = os.path.join(params_path, param_files[model_name])
    try:
        with open(param_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        if yaml_data is None:
            return {}
        return yaml_data
    except FileNotFoundError:
        print(f"경고: 파라미터 파일을 찾을 수 없습니다: {param_file}")
        print("기본 파라미터를 사용합니다.")
        if model_name == "LGBM":
            return {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 20,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "random_state": 42
            }
        elif model_name == "XGB":
            return {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "random_state": 42
            }
