class BaseModel:
    def train(self, X, y):
        raise NotImplementedError

    def evaluate(self, X, y):
        raise NotImplementedError
