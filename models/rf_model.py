from sklearn.ensemble import RandomForestRegressor

class RFModel:
    def __init__(self, params):
        self.params = params
        self.model = RandomForestRegressor(**params)

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)
