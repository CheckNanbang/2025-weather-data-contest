from lightgbm import LGBMRegressor
import numpy as np

class LGBM:
    def __init__(self, params):
        self.model = None
        self.params = params

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        model = LGBMRegressor(**self.params)
        if x_valid is None or y_valid is None:
            model.fit(x_train, y_train)
        else:
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=50,
                eval_metric='rmse',
                verbose=100
            )
        self.model = model
        if x_valid is not None and y_valid is not None:
            y_pred = self.predict(x_valid)
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            print(f"검증 데이터 RMSE: {rmse:.4f}")

    def predict(self, x_data):
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        return self.model.predict(x_data)

    def feature_importance(self):
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        feature_importance = {
            feature: importance
            for feature, importance in zip(
                self.model.feature_name_,
                self.model.feature_importances_
            )
        }
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_importance
