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

        # 결측치 처리
        train_data = train_data.replace(-99, np.nan).dropna(subset=[self.config.data.target_column])
        test_data = test_data.replace(-99, np.nan).dropna(subset=[self.config.data.target_column])

        # 데이터 분할
        split_data = self._split_data(train_data)
        test_features = test_data.drop(columns=["target"], errors="ignore")

        # ✅ 항상 필요한 test_keys 정의
        test_keys = test_data[["tm", "branch_id"]].copy()

        if predict:
            processed_train = self._preprocess_data(train_data, mode='train')
            processed_test = self._preprocess_data(test_data, mode='test')

            split_data = self._split_data(processed_train)
            split_data['x_full'] = processed_train.drop(columns=[self.config.data.target_column])
            split_data['y_full'] = processed_train[self.config.data.target_column]

            # ✅ object → category 변환
            test_features = processed_test.drop(columns=['tm', 'branch_id'], errors='ignore')
            if 'branch_id' in processed_test.columns:
                processed_test['branch_id'] = processed_test['branch_id'].astype('category')
            test_keys = test_keys

        else:
            processed_train = self._preprocess_data(split_data['x_train'], mode='train')
            processed_valid = self._preprocess_data(split_data['x_valid'], mode='valid')
            split_data['x_train'] = processed_train
            split_data['x_valid'] = processed_valid

            test_features = None
            test_keys = None

        # 모델별 학습 및 예측
        for model_name in self.config.training.models:
            self.logger.info(f"클러스터 {self.cluster_id}: {model_name} 모델 학습 시작")
            self._train_single_model(model_name, split_data, test_features, test_keys, predict=predict)

        # 4. 앙상블 (옵션)
        ensemble_result = self._create_ensemble()
        return {
            'predictions': ensemble_result.get('predictions', None),
            'metrics': self.metrics,
            'models': self.models
        }

    def _train_single_model(self, model_name, split_data, test_features, test_keys, predict=False):
        """단일 모델 학습 및 예측"""
        
        model = self.model_factory.create_model(
            model_name,
            cluster_id=self.cluster_id,
            tune_hyperparams=self.config.training.tune_hyperparams
        )

        # 학습
        model.train(split_data['x_train'], split_data['y_train'], split_data['x_valid'], split_data['y_valid'])

        # 평가
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
            final_model = self.model_factory.create_model(
                model_name,
                use_best_params=True,
                cluster_id=self.cluster_id
            )
            final_model.train(split_data['x_full'], split_data['y_full'])

            test_pred = final_model.predict(test_features)
            test_predictions = test_keys.copy()
            test_predictions[self.config.data.target_column] = test_pred

            self.models[model_name] = final_model
            self.predictions[model_name] = test_predictions

            self.file_manager.save_model(final_model, self.cluster_id, self.experiment_id, model_name)
            self.file_manager.save_predictions(test_predictions, self.cluster_id, self.experiment_id, model_name)

        if self.config.logging.save_plots:
            plots = self.evaluator.create_plots(split_data['y_valid'], y_valid_pred, model)
            self.file_manager.save_plots(plots, self.cluster_id, self.experiment_id, model_name)
    
    def _preprocess_data(self, data: pd.DataFrame, mode: str = 'train') -> pd.DataFrame:
        """데이터 전처리: 훈련/테스트 구분"""
        self.logger.info(f"클러스터 {self.cluster_id}: 데이터 전처리 시작")

        if 'Prophet' in self.config.training.models:
            data = data.copy()
            data['ds'] = pd.to_datetime(data['tm'].astype(str), format='%Y%m%d%H', errors='coerce')
            if mode == 'train':
                return data[['ds', 'heat_demand', 'branch_id']]
            else:
                return data[['ds', 'branch_id']]

        keep_cols = ['tm', 'branch_id']
        data_keep = data[keep_cols].reset_index(drop=True)

        if mode == 'train':
            processed = self.preprocessor.fit_transform(data)
        else:
            processed = self.preprocessor.transform(data)

        processed = processed.reset_index(drop=True)

        # 예측 데이터일 경우에만 키 병합
        if mode == 'test':
            return pd.concat([data_keep, processed], axis=1)
        return processed

        

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
            return {'predictions': None}

        if len(self.predictions) == 1:
            return {'predictions': next(iter(self.predictions.values()))}

        self.logger.info(f"클러스터 {self.cluster_id}: 앙상블 생성 중...")

        ensemble_pred = None
        base_df = None

        for model_name, pred_df in self.predictions.items():
            # pred_df = pred_df.sort_values(by='tm').reset_index(drop=True)

            if ensemble_pred is None:
                ensemble_pred = pred_df[self.config.data.target_column].values
                base_df = pred_df[['tm']].copy()
            else:
                ensemble_pred += pred_df[self.config.data.target_column].values

        ensemble_pred = ensemble_pred / len(self.predictions)

        final_predictions = base_df.copy()
        final_predictions[self.config.data.target_column] = ensemble_pred

        return {'predictions': final_predictions}

    
    def tune(self, train_data: pd.DataFrame) -> dict:
        self.logger.info(f"클러스터 {self.cluster_id}: 하이퍼파라미터 튜닝 시작")

        processed_train = self.preprocessor.transform(train_data)
        split_data = self._split_data(processed_train)

        x_train, y_train = split_data['x_train'], split_data['y_train']
        x_valid, y_valid = split_data['x_valid'], split_data['y_valid']

        # train에서 이상치 제거
        invalid_train_idx = y_train[
            y_train.isnull() | np.isinf(y_train) | (y_train > 1e10)
        ].index
        if len(invalid_train_idx) > 0:
            self.logger.warning(f"y_train에서 {len(invalid_train_idx)}개의 NaN/무한대/과도한 값 발견, 해당 행 제거")
        x_train = x_train.drop(index=invalid_train_idx)
        y_train = y_train.drop(index=invalid_train_idx)

        # valid에서 이상치 제거
        invalid_valid_idx = y_valid[
            y_valid.isnull() | np.isinf(y_valid) | (y_valid > 1e10)
        ].index
        if len(invalid_valid_idx) > 0:
            self.logger.warning(f"y_valid에서 {len(invalid_valid_idx)}개의 NaN/무한대/과도한 값 발견, 해당 행 제거")
        x_valid = x_valid.drop(index=invalid_valid_idx)
        y_valid = y_valid.drop(index=invalid_valid_idx)

        best_params_dict = {}

        for model_name in self.config.training.models:
            self.logger.info(f"클러스터 {self.cluster_id}: {model_name} 튜닝 시작")

            best_params = self.model_factory.tune_hyperparameters(
                model_name, x_train, y_train, x_valid, y_valid, self.cluster_id
            )

            if best_params is not None:
                self.logger.info(f"{model_name} best_params: {best_params}")
                best_params_dict[model_name] = best_params
            else:
                self.logger.warning(f"{model_name} 튜닝 결과를 찾을 수 없습니다.")

        return best_params_dict
