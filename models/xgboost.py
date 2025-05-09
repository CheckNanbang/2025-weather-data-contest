from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train(X_train, y_train, X_test, y_test, params):
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse
