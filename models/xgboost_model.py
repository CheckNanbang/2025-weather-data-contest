import xgboost as xgb

class XGBoostModel:
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBRegressor(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)
