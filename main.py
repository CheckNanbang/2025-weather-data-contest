# main.py
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Tuple, Any

# 로컬 모듈 임포트
from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_factory import ModelFactory
from src.training.trainer import ClusterTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logging
from src.utils.config import Config
from src.utils.file_manager import FileManager

class ClusterMLPipeline:
    """클러스터별 머신러닝 파이프라인 메인 클래스"""

    def __init__(self, config_path: str = "config/config.yaml"):
        # 설정 및 각종 유틸 객체 초기화
        self.config = Config(config_path)
        self.logger = setup_logging(self.config)
        self.file_manager = FileManager(self.config)
        self.experiment_id = self._generate_experiment_id()
        
        # 클러스터별 결과 저장용 딕셔너리
        self.cluster_results = {}
        self.cluster_metrics = {}
        
        # 전처리 객체 저장 (클러스터별 공유)
        self.preprocessors = {}

    def _generate_experiment_id(self) -> str:
        """실험 ID 생성 (실험명_날짜시간)"""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return f"{self.config.experiment.name}_{timestamp}"

    def run(self, selected_clusters: List[int] = None, predict: bool = False) -> Dict[str, Any]:
        """파이프라인 전체 실행 함수"""
        self.logger.info(f"🚀 실험 시작: {self.experiment_id}")
        try:
            # 데이터 로딩
            train_df, test_df, submission_df = self._load_data()
            print(f"학습 데이터 크기: {len(train_df)}, 테스트 데이터 크기: {len(test_df)}")
            
            
            # branch_id → cluster_id 매핑 및 month 컬럼 생성
            train_df, test_df = self._add_cluster_ids(train_df, test_df)
            
            # 클러스터 미지정 시 전체 사용
            if selected_clusters is None:
                selected_clusters = list(self.config.cluster.mapping.keys())
            
            # 클러스터별 학습/예측
            self._train_clusters(train_df, test_df, selected_clusters, predict=predict)
            
            submission_file = None
            # 예측 모드면 제출 파일 생성
            if predict:
                submission_file = self._generate_submission(submission_df)
            
            # 결과 저장 및 리포트 생성
            self._save_results_and_report()
            self.logger.info("✅ 실험 완료!")
            
            return {
                'experiment_id': self.experiment_id,
                'cluster_metrics': self.cluster_metrics,
                'submission_file': submission_file
            }
        except Exception as e:
            self.logger.error(f"❌ 실험 실패: {str(e)}", exc_info=True)
            raise

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로딩 함수"""
        self.logger.info("📊 데이터 로딩 중...")
        data_loader = DataLoader(self.config)
        return data_loader.load()

    def _add_cluster_ids(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """branch_id를 cluster_id로 매핑하여 컬럼 추가 및 month 컬럼 생성"""
        self.logger.info("🏷️ 클러스터 ID 매핑 중...")
        
        # branch_id → cluster_id 매핑 딕셔너리 생성
        branch_to_cluster = {}
        for cluster_id, branches in self.config.cluster.mapping.items():
            for branch in branches:
                branch_to_cluster[branch] = cluster_id
        
        # 매핑 적용
        train_df['cluster_id'] = train_df['branch_id'].map(branch_to_cluster)
        test_df['cluster_id'] = test_df['branch_id'].map(branch_to_cluster)
        
        # month 컬럼 생성 (데이터 타입 처리 개선)
        try:
            train_df['month'] = pd.to_datetime(train_df['tm'], errors='coerce', format='%Y%m%d%H').dt.month
            test_df['month'] = pd.to_datetime(test_df['tm'], errors='coerce', format='%Y%m%d%H').dt.month
            
            # NaN 값을 -1로 채우고 정수형으로 변환
            train_df['month'] = train_df['month'].fillna(-1).astype(int)
            test_df['month'] = test_df['month'].fillna(-1).astype(int)
            
        except KeyError as e:
            self.logger.error(f"날짜(tm) 컬럼이 존재하지 않습니다: {e}")
            raise   
        
        # 매핑되지 않은 데이터 확인
        unmapped_train = train_df[train_df['cluster_id'].isna()]
        unmapped_test = test_df[test_df['cluster_id'].isna()]
        if len(unmapped_train) > 0 or len(unmapped_test) > 0:
            self.logger.warning(f"매핑되지 않은 데이터: train {len(unmapped_train)}, test {len(unmapped_test)}")
        
        return train_df, test_df

    def _prepare_cluster2_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Cluster 2 데이터를 summer/rest로 분할"""
        self.logger.info("🔄 Cluster 2 데이터 분할 중...")
        
        # Cluster 2 데이터 필터링
        train_cluster2 = train_df[train_df['cluster_id'] == 2].copy()
        test_cluster2 = test_df[test_df['cluster_id'] == 2].copy()
        
        # 월별 데이터 개수 확인
        self.logger.info(f"Cluster 2 월별 분포:\n{train_cluster2['month'].value_counts().sort_index()}")
        
        # 여름(6~9월)과 나머지 분할
        summer_months = [6, 7, 8, 9]
        
        train_summer = train_cluster2[train_cluster2['month'].isin(summer_months)].copy()
        train_rest = train_cluster2[~train_cluster2['month'].isin(summer_months)].copy()
        test_summer = test_cluster2[test_cluster2['month'].isin(summer_months)].copy()
        test_rest = test_cluster2[~test_cluster2['month'].isin(summer_months)].copy()
        
        self.logger.info(f"Cluster 2 분할 결과: 여름 {len(train_summer)}개, 나머지 {len(train_rest)}개")
        
        return {
            'train_summer': train_summer,
            'train_rest': train_rest,
            'test_summer': test_summer,
            'test_rest': test_rest
        }

    # def _preprocess_cluster_data(self, cluster_id: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """클러스터별 데이터 전처리 (fit_transform → transform)"""
    #     self.logger.info(f"🔧 Cluster {cluster_id} 전처리 시작")
        
    #     # 전처리 객체가 없으면 새로 생성
    #     if cluster_id not in self.preprocessors:
    #         self.preprocessors[cluster_id] = DataPreprocessor(self.config)
        
    #     preprocessor = self.preprocessors[cluster_id]
        
    #     # 학습 데이터로 fit_transform
    #     train_processed = preprocessor.fit_transform(train_data)
        
    #     # 테스트 데이터는 transform만
    #     test_processed = preprocessor.transform(test_data)
        
    #     self.logger.info(f"✅ Cluster {cluster_id} 전처리 완료: train {train_processed.shape}, test {test_processed.shape}")
        
    #     return train_processed, test_processed

    def _train_single_cluster(self, cluster_id: str, train_data: pd.DataFrame, test_data: pd.DataFrame, predict: bool = False) -> Dict[str, Any]:
        """단일 클러스터/서브그룹 학습"""
        if len(train_data) == 0:
            self.logger.warning(f"Cluster {cluster_id}에 학습 데이터가 없습니다. 건너뜁니다.")
            return {'predictions': None, 'metrics': {}}
        
        # # 전처리 수행
        # train_processed, test_processed = self._preprocess_cluster_data(cluster_id, train_data, test_data)
        
        # 모델 학습
        cluster_trainer = ClusterTrainer(self.config, cluster_id, self.experiment_id)
        result = cluster_trainer.train_and_predict(train_data, test_data, predict=predict)
        
        return result

    def _train_clusters(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_clusters: List[int],
        predict=False
    ) -> None:
        self.logger.info(f"🎯 선택된 클러스터: {selected_clusters}")

        all_valid_true = []
        all_valid_pred = []

        for cluster_id in selected_clusters:
            self.logger.info(f"{'='*20} 클러스터 {cluster_id} 처리 시작 {'='*20}")

            # Cluster 2만 특별 처리
            if (str(cluster_id) == "2" and 
                hasattr(self, "cluster2_mode") and 
                self.cluster2_mode == "split"):
                
                # Cluster 2 데이터 분할
                cluster2_data = self._prepare_cluster2_data(train_df, test_df)
                
                # Summer 그룹 학습
                if len(cluster2_data['train_summer']) > 0:
                    self.logger.info("🌞 Cluster 2_summer (6~9월) 학습")
                    result = self._train_single_cluster(
                        "2_summer", 
                        cluster2_data['train_summer'], 
                        cluster2_data['test_summer'], 
                        predict=predict
                    )
                    
                    if result.get('predictions') is not None:
                        self.cluster_results["2_summer"] = result['predictions']
                    self.cluster_metrics["2_summer"] = result['metrics']
                    
                    # 검증 결과 집계
                    self._collect_validation_results(result, all_valid_true, all_valid_pred)

                # Rest 그룹 학습
                if len(cluster2_data['train_rest']) > 0:
                    self.logger.info("❄️ Cluster 2_rest (1~5,10~12월) 학습")
                    result = self._train_single_cluster(
                        "2_rest", 
                        cluster2_data['train_rest'], 
                        cluster2_data['test_rest'], 
                        predict=predict
                    )
                    
                    if result.get('predictions') is not None:
                        self.cluster_results["2_rest"] = result['predictions']
                    self.cluster_metrics["2_rest"] = result['metrics']
                    
                    # 검증 결과 집계
                    self._collect_validation_results(result, all_valid_true, all_valid_pred)
                
                continue  # 다른 클러스터 처리로 넘어감

            # 나머지 클러스터 일반 처리
            train_cluster = train_df[train_df['cluster_id'] == cluster_id].copy()
            test_cluster = test_df[test_df['cluster_id'] == cluster_id].copy()
            print(f"클러스터 {cluster_id} 학습 데이터 크기: {len(train_cluster)}")
            print(f"클러스터 {cluster_id} 테스트 데이터 크기: {len(test_cluster)}")

            result = self._train_single_cluster(str(cluster_id), train_cluster, test_cluster, predict=predict)
            
            if result.get('predictions') is not None:
                self.cluster_results[cluster_id] = result['predictions']
            self.cluster_metrics[cluster_id] = result['metrics']

            self.logger.info(f"✅ 클러스터 {cluster_id} 완료")
            
            # 검증 결과 집계
            self._collect_validation_results(result, all_valid_true, all_valid_pred)

        # 전체(글로벌) 평가지표 계산
        self._calculate_global_metrics(all_valid_true, all_valid_pred)

    def _collect_validation_results(self, result: Dict[str, Any], all_valid_true: List, all_valid_pred: List) -> None:
        """검증 결과 수집"""
        best_model = None
        best_rmse = float('inf')
        best_metrics = None
        
        for model_name, metrics in result['metrics'].items():
            rmse = metrics.get("RMSE")
            if rmse is not None and rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
                best_metrics = metrics
        
        if best_metrics is not None:
            y_valid_true = best_metrics.get("y_valid_true")
            y_valid_pred = best_metrics.get("y_valid_pred")
            if y_valid_true is not None and y_valid_pred is not None:
                all_valid_true.append(y_valid_true)
                all_valid_pred.append(y_valid_pred)

    def _calculate_global_metrics(self, all_valid_true: List, all_valid_pred: List) -> None:
        """전체 평가지표 계산"""
        self.global_metrics = None
        if all_valid_true and all_valid_pred:
            y_true_all = np.concatenate(all_valid_true)
            y_pred_all = np.concatenate(all_valid_pred)
            evaluator = ModelEvaluator(self.config)
            self.global_metrics = evaluator.evaluate(y_true_all, y_pred_all)
            
            self.logger.info("====== 전체(글로벌) 검증 평가지표 (클러스터별 베스트 모델 기준) ======")
            for metric, value in self.global_metrics.items():
                if value is not None:
                    self.logger.info(f"  {metric}: {value:.4f}")

    def _save_results_and_report(self) -> None:
        """결과 저장 및 리포트 생성 함수"""
        self.logger.info("💾 결과 저장 및 리포트 생성 중...")
        # 메트릭 저장
        self.file_manager.save_metrics(self.cluster_metrics, self.experiment_id)
        # 종합 리포트 생성
        evaluator = ModelEvaluator(self.config)
        evaluator.generate_report(
            self.cluster_metrics, 
            self.experiment_id, 
            global_metrics=getattr(self, "global_metrics", None)
        )

    def _generate_submission(self, submission_df: pd.DataFrame) -> str:
        """클러스터별 예측 결과를 통합하여 제출 파일 생성"""
        if not self.cluster_results:
            self.logger.warning("예측 결과가 없어 제출 파일을 생성할 수 없습니다.")
            return None
            
        self.logger.info("📝 제출 파일 생성 중...")
        
        # 클러스터별 예측 결과 통합
        all_predictions = []
        for cluster_id, predictions in self.cluster_results.items():
            all_predictions.append(predictions)
        
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        final_predictions = final_predictions.sort_values('id')
        
        # 제출용 데이터프레임과 병합
        submission_df = submission_df.drop(columns=[self.config.data.target_column], errors="ignore")
        submission_df = submission_df.merge(
            final_predictions[['id', self.config.data.target_column]],
            on='id', how='left'
        )
        
        # 파일 저장
        submission_file = self.file_manager.save_submission(submission_df, self.experiment_id)
        self.logger.info(f"📁 제출 파일 저장: {submission_file}")
        return submission_file

    def tune(self, selected_clusters: List[int] = None):
        """클러스터별 하이퍼파라미터 튜닝만 수행"""
        self.logger.info(f"🛠️ 튜닝 시작: {self.experiment_id}")
        train_df, test_df, _ = self._load_data()
        train_df, test_df = self._add_cluster_ids(train_df, test_df)
        
        if selected_clusters is None:
            selected_clusters = list(self.config.cluster.mapping.keys())
        
        for cluster_id in selected_clusters:
            self.logger.info(f"클러스터 {cluster_id} 튜닝 시작")
            train_cluster = train_df[train_df['cluster_id'] == cluster_id].copy()
            
            if len(train_cluster) == 0:
                self.logger.warning(f"클러스터 {cluster_id}에 학습 데이터가 없습니다. 건너뜁니다.")
                continue
            
            cluster_trainer = ClusterTrainer(self.config, cluster_id, self.experiment_id)
            # 1. 전처리기 fit 먼저 수행 (train_cluster는 원본 데이터)
            cluster_trainer.preprocessor.fit_transform(train_cluster)

            # 2. 이후 tune에서는 transform만 하도록 변경된 tune 함수 호출
            best_params = cluster_trainer.tune(train_cluster)
            self.logger.info(f"클러스터 {cluster_id} 튜닝 결과: {best_params}")
        
        self.logger.info("✅ 모든 클러스터 튜닝 완료")

def main():
    """메인 함수: 커맨드라인 인자 파싱 및 파이프라인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="클러스터별 ML 파이프라인")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="설정 파일 경로")
    parser.add_argument("--clusters", nargs="*", type=int,
                       help="학습할 클러스터 ID (기본값: 모든 클러스터)")
    parser.add_argument(
        "--cluster2-mode",
        choices=["split", "all"],
        default="split",
        help="cluster 2 학습 방식: split(여름/비여름 분리, 기본값) 또는 all(전체 학습)"
    )
    parser.add_argument("--models", nargs="+", help="학습할 모델명 리스트 (예: --models LGBM XGB)")
    parser.add_argument("--predict", action="store_true", help="최종 예측(제출)까지 실행")
    parser.add_argument("--tune", action="store_true", help="하이퍼파라미터 튜닝만 실행")
    
    args = parser.parse_args()
    
    # 파이프라인 객체 생성 및 옵션 반영
    pipeline = ClusterMLPipeline(args.config)
    pipeline.cluster2_mode = args.cluster2_mode
    
    if args.models:
        pipeline.config.training.models = args.models
    
    # 튜닝만 수행
    if args.tune:
        pipeline.tune(selected_clusters=args.clusters)
        return
    
    # 학습/예측 실행
    results = pipeline.run(selected_clusters=args.clusters, predict=args.predict)
    print(f"\n🎉 실험 완료! ID: {results['experiment_id']}")

if __name__ == "__main__":
    main()