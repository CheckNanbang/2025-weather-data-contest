from prophet import Prophet
from .base_model import BaseModel
import pandas as pd

class ProphetModel(BaseModel):
    def __init__(self, params):
        super().__init__(params)
        # Prophet 공식 문서의 허용 파라미터만 추출
        valid_params = {
            k: v for k, v in params.items() 
            if k in Prophet.__init__.__code__.co_varnames
        }
        self.model = Prophet(**valid_params)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        print("Prophet 입력 샘플:")
        print(x_train.head())
        print("Prophet 타겟 샘플:")
        print(y_train.head())
        df = pd.DataFrame({'ds': x_train['ds'], 'y': y_train})
        self.model.fit(df)

    def predict(self, x_data):
        future = self.model.make_future_dataframe(
            periods=len(x_data), 
            include_history=False
        )
        return self.model.predict(future)['yhat'].values

    def get_feature_importance(self):
        return {}  # Prophet은 특성 중요도 제공 안함
