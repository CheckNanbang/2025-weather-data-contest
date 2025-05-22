def train_model(model_name, params, x_train, y_train, x_valid=None, y_valid=None):
    supported_models = ["LGBM", "XGB"]

    if model_name not in supported_models:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}. 지원되는 모델: {supported_models}")

    if model_name == "LGBM":
        from models.lgbm import LGBM
        model = LGBM(params)
    elif model_name == "XGB":
        from models.xgb import XGB
        model = XGB(params)

    model.train(x_train, y_train, x_valid, y_valid)
    return model
