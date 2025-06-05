import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

from ..data.preprocessor import DataPreprocessor
from ..models.model_factory import ModelFactory
from ..evaluation.evaluator import ModelEvaluator
from ..utils.file_manager import FileManager

class ClusterTrainer:
    """클러스터별 학습을 담당하는 클래스"""

    def __init__(self, config: Any, cluster_id: int, experiment_id: str):
        self.config = config
        self.cluster_id = cluster_id
        self.experiment_id = experiment_id
        self.logger = logging.getLogger('cluster_ml')

        # 컴포넌트 초기화
        self.preprocessor = DataPreprocessor(config)
        self.model_factory = ModelFactory(config)
        self.evaluator = ModelEvaluator(config)
        self.file_manager = FileManager(config)

        # 클러스터별 결과 저장
        self.models = {}
        self.predictions = {}
        self.metrics = {}

    def train_and_predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame, predict: bool = False) -> Dict[str, Any]:
        """클러스터별 전체 학습 및 예측 파이프라인"""
        self.logger.info(f"클러스터 {self.cluster_id}: 데이터 크기 - 학습: {len(train_data)}, 테스트: {len(test_data)}")

        # 0. target에서 결측치 제거
        train_data = train_data.replace(-99, np.nan)
        test_data = test_data.replace(-99, np.nan)
        train_data = train_data.dropna(subset=[self.config.data.target_column])
        test_data = test_data.dropna(subset=[self.config.data.target_column])
        
        # 1. 데이터 분할
        split_data = self._split_data(train_data)
        
        # # 2. 데이터 전처리
        processed_data = self._preprocess_data(split_data['x_train'], split_data['x_valid'])
        split_data['x_train'] = processed_data['train']
        split_data['x_valid'] = processed_data['test']

        # 전처리 없이 코드 돌리려면 아래 두 줄 주석 해제
        # split_data['x_train'].drop(columns=['tm', 'branch_id'], inplace=True)
        # split_data['x_valid'].drop(columns=['tm', 'branch_id'], inplace=True)
        
        # 3. 모델별 학습 및 예측
        for model_name in self.config.training.models:
            self.logger.info(f"클러스터 {self.cluster_id}: {model_name} 모델 학습 시작")
            self._train_single_model(model_name, split_data, test_data, predict=predict)

        print(f"DEBUG: self.metrics in train_and_predict (cluster {self.cluster_id}) =", self.metrics)
        
        # 4. 앙상블 (옵션)
        ensemble_result = self._create_ensemble()
        return {
            'predictions': ensemble_result.get('predictions', None),
            'metrics': self.metrics,
            'models': self.models
        }

    def _train_single_model(self, model_name: str, split_data: Dict[str, Any], test_data: pd.DataFrame, predict: bool = False) -> None:
        """단일 모델 학습 및 예측"""

        # 1. 모델 생성 및 하이퍼파라미터 튜닝
        model = self.model_factory.create_model(
            model_name,
            cluster_id=self.cluster_id,
            tune_hyperparams=self.config.training.tune_hyperparams
        )

        # 2. 모델 학습 (검증 데이터 포함)
        model.train(
            split_data['x_train'],
            split_data['y_train'],
            split_data['x_valid'],
            split_data['y_valid']
        )

        # 3. 검증 성능 평가
        y_valid_pred = model.predict(split_data['x_valid'])
        validation_metrics = self.evaluator.evaluate(split_data['y_valid'], y_valid_pred)

        self.logger.info(f"클러스터 {self.cluster_id} {model_name} 검증 성능:")
        for metric, value in validation_metrics.items():
            if value is not None:
                self.logger.info(f"  {metric}: {value:.4f}")

        self.metrics[model_name] = {
            **validation_metrics,
            "y_valid_true": split_data['y_valid'].to_numpy(),
            "y_valid_pred": y_valid_pred
        }

        if predict:
            # 4. 전체 데이터로 재학습 (최종 모델)
            
            # TODO
            # 전체로 전처리 하는 코드가 있어야 함!!!!!!!!!!!!!!!!!!!!!!!!!
            
            final_model = self.model_factory.create_model(model_name, use_best_params=True)
            final_model.train(split_data['x_full'], split_data['y_full'])

            # 5. 테스트 예측
            test_features = test_data.drop(columns=[self.config.data.target_column], errors='ignore')
            test_pred = final_model.predict(test_features)

            # 예측 결과 저장
            test_predictions = test_data[['id']].copy()
            test_predictions[self.config.data.target_column] = test_pred

            # 6. 결과 저장
            self.models[model_name] = final_model
            self.predictions[model_name] = test_predictions

            # 7. 파일 저장
            self.file_manager.save_model(final_model, self.cluster_id, self.experiment_id, model_name)
            self.file_manager.save_predictions(test_predictions, self.cluster_id, self.experiment_id, model_name)

        # 8. 시각화 (옵션)
        if self.config.logging.save_plots:
            plots = self.evaluator.create_plots(
                split_data['y_valid'],
                y_valid_pred,
                model
            )
            self.file_manager.save_plots(plots, self.cluster_id, self.experiment_id, model_name)

    def _preprocess_data(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """데이터 전처리"""
        self.logger.info(f"클러스터 {self.cluster_id}: 데이터 전처리 시작")

        # 전처리 수행 (구체적인 구현은 preprocessor에서)        
        if 'Prophet' in self.config.training.models:
            """Prophet 모델 전용 전처리 분기"""
            train_data['ds'] = pd.to_datetime(
                train_data['tm'].astype(str), 
                format='%Y%m%d%H',
                errors='coerce'
            )
            test_data['ds'] = pd.to_datetime(
                test_data['tm'].astype(str),
                format='%Y%m%d%H',
                errors='coerce'
            )
            return {
                'train': train_data[['ds', 'heat_demand', 'branch_id']],
                'test': test_data[['ds', 'branch_id']]
            }
        else:
            processed_train = self.preprocessor.fit_transform(train_data)
            processed_test = self.preprocessor.transform(test_data)
                      
            self.logger.info(f"클러스터 {self.cluster_id}: 전처리 완료 - 학습: {processed_train.shape}, 테스트: {processed_test.shape}")

            return {
                'train': processed_train,
                'test': processed_test
            }


    def _split_data(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        """데이터 분할"""
        target_col = self.config.data.target_column

        if self.config.split.type == "time":
            # 시간 기반 분할
            start_time = self.config.split.time_split['holdout_start']
            end_time = self.config.split.time_split['holdout_end']

            is_holdout = ((train_data["tm"] >= start_time) &
                         (train_data["tm"] <= end_time))

            x_train = train_data[~is_holdout].drop(columns=[target_col])
            y_train = train_data[~is_holdout][target_col]
            x_valid = train_data[is_holdout].drop(columns=[target_col])
            y_valid = train_data[is_holdout][target_col]

        else:  # random split
            from sklearn.model_selection import train_test_split

            X = train_data.drop(columns=[target_col])
            y = train_data[target_col]

            x_train, x_valid, y_train, y_valid = train_test_split(
                X, y,
                test_size=self.config.split.test_size,
                random_state=self.config.split.random_state
            )

        self.logger.info(f"클러스터 {self.cluster_id}: 데이터 분할 완료 - 학습: {len(x_train)}, 검증: {len(x_valid)}")

        return {
            'x_train': x_train,
            'y_train': y_train,
            'x_valid': x_valid,
            'y_valid': y_valid,
            'x_full': train_data.drop(columns=[target_col]),
            'y_full': train_data[target_col]
        }

    def _create_ensemble(self) -> Dict[str, Any]:
        """앙상블 모델 생성 (단순 평균)"""
        if not self.predictions:
            # 예측 결과가 없으면 None 반환
            return {'predictions': None}

        if len(self.predictions) == 1:
            model_name = list(self.predictions.keys())[0]
            return {'predictions': self.predictions[model_name]}

        self.logger.info(f"클러스터 {self.cluster_id}: 앙상블 생성 중...")

        # 모든 예측 결과의 평균 계산
        ensemble_pred = None
        base_df = None

        for model_name, pred_df in self.predictions.items():
            if ensemble_pred is None:
                ensemble_pred = pred_df[self.config.data.target_column].values
                base_df = pred_df[['id']].copy()
            else:
                ensemble_pred += pred_df[self.config.data.target_column].values

        ensemble_pred = ensemble_pred / len(self.predictions)

        # 최종 예측 결과 생성
        final_predictions = base_df.copy()
        final_predictions[self.config.data.target_column] = ensemble_pred

        return {'predictions': final_predictions}
    
    def tune(self, train_data: pd.DataFrame) -> dict:
        """클러스터별 하이퍼파라미터 튜닝만 수행하고 결과 반환"""
        self.logger.info(f"클러스터 {self.cluster_id}: 하이퍼파라미터 튜닝 시작")
        processed_train = self.preprocessor.fit_transform(train_data)
        split_data = self._split_data(processed_train)
        best_params_dict = {}

        for model_name in self.config.training.models:
            self.logger.info(f"클러스터 {self.cluster_id}: {model_name} 튜닝 시작")
            x_train, y_train = split_data['x_train'], split_data['y_train']
            x_valid, y_valid = split_data['x_valid'], split_data['y_valid']

            # ★ 여기서 실제 튜닝 실행!
            best_params = self.model_factory.tune_hyperparameters(
                model_name, x_train, y_train, x_valid, y_valid, self.cluster_id
            )
            if best_params is not None:
                self.logger.info(f"{model_name} best_params: {best_params}")
                best_params_dict[model_name] = best_params
            else:
                self.logger.warning(f"{model_name} 튜닝 결과를 찾을 수 없습니다.")
        return best_params_dict
