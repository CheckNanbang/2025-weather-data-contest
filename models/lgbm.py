from lightgbm import LGBMRegressor

class LGBM:
    """
    LightGBM 회귀 모델을 다루는 클래스.
    
    Args:
        params (dict): LightGBM 모델 학습에 필요한 하이퍼파라미터
    """
    def __init__(self, params):
        self.model = None
        self.params = params
        
    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        LightGBM 회귀 모델을 학습시킴.
        
        Args:
            x_train (pd.DataFrame): 학습 입력 데이터
            y_train (pd.Series): 학습 타겟 데이터
            x_valid (pd.DataFrame, optional): 검증 입력 데이터
            y_valid (pd.Series, optional): 검증 타겟 데이터
        """
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
        
        # 성능 평가 (검증 데이터가 있는 경우)
        if x_valid is not None and y_valid is not None:
            y_pred = self.predict(x_valid)
            from sklearn.metrics import mean_squared_error
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            print(f"검증 데이터 RMSE: {rmse:.4f}")
    
    def predict(self, x_data):
        """
        학습된 모델을 사용하여 예측 수행.
        
        Args:
            x_data (pd.DataFrame): 예측할 입력 데이터
            
        Returns:
            np.ndarray: 예측된 값
            
        Raises:
            ValueError: 모델이 학습되지 않은 경우
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        return self.model.predict(x_data)
    
    def feature_importance(self):
        """
        학습된 모델의 특성 중요도를 반환.
        
        Returns:
            dict: 특성 이름과 중요도를 포함하는 딕셔너리
            
        Raises:
            ValueError: 모델이 학습되지 않은 경우
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train() 메서드를 호출하세요.")
        
        # 특성 이름과 중요도를 매핑
        feature_importance = {
            feature: importance 
            for feature, importance in zip(
                self.model.feature_name_,
                self.model.feature_importances_
            )
        }
        
        # 중요도 기준으로 정렬
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
        )
        
        return sorted_importance