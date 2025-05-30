# src/models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

class BaseModel(ABC):
    """모델 베이스 클래스"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, x_train: pd.DataFrame, y_train: pd.Series, 
              x_valid: Optional[pd.DataFrame] = None, 
              y_valid: Optional[pd.Series] = None) -> None:
        """모델 학습"""
        pass
    
    @abstractmethod  
    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 반환"""
        pass