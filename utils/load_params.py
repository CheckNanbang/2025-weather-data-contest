import os
import yaml

def load_params(model_name):
    """
    특정 모델에 맞는 회귀 하이퍼파라미터를 YAML 파일에서 불러옴.

    Args:
        model_name (str): 불러올 모델 이름, "LGBM", "XGB" 중 하나여야 함

    Returns:
        dict: 모델에 맞는 회귀 하이퍼파라미터 딕셔너리
        
    Raises:
        ValueError: 지원하지 않는 모델 이름이 제공될 경우
    """
    supported_models = ["LGBM", "XGB"]
    params_path = os.path.join("models", "params")
    
    if model_name not in supported_models:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}. 지원되는 모델: {supported_models}")
    
    # 모델 이름에 맞는 파라미터 파일 로드
    param_files = {
        "LGBM": "lgbm_param.yaml",
        "XGB": "xgb_param.yaml"
    }
    
    param_file = os.path.join(params_path, param_files[model_name])
    
    try:
        with open(param_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        # yaml_data가 None이면 빈 dict 반환
        if yaml_data is None:
            return {}
        return yaml_data
    except FileNotFoundError:
        print(f"경고: 파라미터 파일을 찾을 수 없습니다: {param_file}")
        print("기본 파라미터를 사용합니다.")
        
        # 기본 파라미터 설정
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
