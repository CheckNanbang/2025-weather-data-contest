# models/train_model.py
def train_model(model_name, params, x_train, y_train, x_valid=None, y_valid=None):
    """
    주어진 모델에 맞는 회귀 모델을 학습시킴.

    Args:
        model_name (str): 학습할 모델 이름, "LGBM", "XGB" 중 하나
        params (dict): 모델 학습에 사용할 하이퍼파라미터
        x_train (pd.DataFrame): 학습에 사용할 입력 데이터
        y_train (pd.Series): 학습에 사용할 타겟 데이터
        x_valid (pd.DataFrame, optional): 검증용 입력 데이터
        y_valid (pd.Series, optional): 검증용 타겟 데이터

    Returns:
        model: 학습된 모델 객체

    Raises:
        ValueError: 지원하지 않는 모델 이름이 제공될 경우
    """
    supported_models = ["LGBM", "XGB"]

    if model_name not in supported_models:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}. 지원되는 모델: {supported_models}")
    
    if model_name == "LGBM":
        from models.lgbm import LGBM
        model = LGBM(params)
    elif model_name == "XGB":
        from models.xgb import XGB
        model = XGB(params)
    
    # 검증 데이터가 제공된 경우 함께 학습
    model.train(x_train, y_train, x_valid, y_valid)

    return model