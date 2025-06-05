# src/models/model_factory.py
import optuna
import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.metrics import mean_squared_error

from .lgbm_model import LGBMModel
from .xgb_model import XGBModel
from .prophet_model import ProphetModel

class ModelFactory:
    """모델 생성 팩토리 클래스"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('cluster_ml')
        self.best_params = {}  # 클러스터별 최적 파라미터 저장
        
    def create_model(self, model_name: str, cluster_id: Optional[int] = None, 
                    tune_hyperparams: bool = False, use_best_params: bool = False):
        """모델 인스턴스 생성"""
        
        # 기본 파라미터 로드
        if model_name == "LGBM":
            model_class = LGBMModel
            default_params = self._get_lgbm_default_params()
        elif model_name == "XGB":
            model_class = XGBModel  
            default_params = self._get_xgb_default_params()
        elif model_name == "Prophet":
            model_class = ProphetModel
            default_params = self._get_prophet_default_params()
        else:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        # 최적 파라미터 사용
        if use_best_params and cluster_id is not None:
            key = f"{model_name}_{cluster_id}"
            if key in self.best_params:
                params = {**default_params, **self.best_params[key]}
            else:
                params = default_params
        else:
            params = default_params
            
        return model_class(params)
    
    def tune_hyperparameters(self, model_name: str, x_train, y_train, 
                           x_valid, y_valid, cluster_id: int) -> Dict[str, Any]:
        """하이퍼파라미터 튜닝"""
        self.logger.info(f"클러스터 {cluster_id} {model_name} 하이퍼파라미터 튜닝 시작")
        
        def objective(trial):
            if model_name == "LGBM":
                params = self._suggest_lgbm_params(trial)
            elif model_name == "XGB":
                params = self._suggest_xgb_params(trial)
            else:
                raise ValueError(f"튜닝을 지원하지 않는 모델: {model_name}")
            
            # 모델 학습 및 평가
            model = self.create_model(model_name)
            model.params.update(params)
            model.train(x_train, y_train, x_valid, y_valid)
            
            y_pred = model.predict(x_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
            
            return rmse
        
        # Optuna 스터디 실행
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.training.n_trials)
        
        # 최적 파라미터 저장
        best_params = study.best_trial.params
        key = f"{model_name}_{cluster_id}"
        self.best_params[key] = best_params
        
        self.logger.info(f"클러스터 {cluster_id} {model_name} 최적 파라미터:")
        for param, value in best_params.items():
            self.logger.info(f"  {param}: {value}")
        
        return best_params
    
    def _get_lgbm_default_params(self) -> Dict[str, Any]:
        """LightGBM 기본 파라미터"""
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def _get_xgb_default_params(self) -> Dict[str, Any]:
        """XGBoost 기본 파라미터"""
        return {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
    
    def _get_prophet_default_params(self) -> Dict[str, Any]:
        """Prophet 기본 파라미터"""
        return {
            'growth': 'linear',
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    
    def _suggest_lgbm_params(self, trial) -> Dict[str, Any]:
        """LightGBM 파라미터 제안"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
    
    def _suggest_xgb_params(self, trial) -> Dict[str, Any]:
        """XGBoost 파라미터 제안"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }

    def _get_prophet_default_params(self) -> Dict[str, Any]:
        from prophet import Prophet
        return {
            k: v for k, v in self.config.training.prophet_params.items()
            if k in Prophet.__init__.__code__.co_varnames
        }
