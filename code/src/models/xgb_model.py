# src/models/xgb_model.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
import xgboost as xgb
from .base_model import BaseModel

class XGBModel(BaseModel):
    """XGBoost 모델 클래스"""
    
    def train(self, x_train: pd.DataFrame, y_train: pd.Series,
              x_valid: Optional[pd.DataFrame] = None,
              y_valid: Optional[pd.Series] = None) -> None:
        """모델 학습"""
        
        model = xgb.XGBRegressor(**self.params)
        
        if x_valid is not None and y_valid is not None:
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=100
            )
        else:
            model.fit(x_train, y_train)
        
        self.model = model
        self.is_trained = True
    
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.model.predict(x_data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 반환"""
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

