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
        self.config = Config(config_path)
        self.logger = setup_logging(self.config)
        self.file_manager = FileManager(self.config)
        self.experiment_id = self._generate_experiment_id()
        
        # 결과 저장용 딕셔너리
        self.cluster_results = {}
        self.cluster_metrics = {}
        
    def _generate_experiment_id(self) -> str:
        """실험 ID 생성"""
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        return f"{self.config.experiment.name}_{timestamp}"
    
    def run(self, selected_clusters: List[int] = None, predict: bool = False) -> Dict[str, Any]:
        """메인 실행 함수"""
        self.logger.info(f"🚀 실험 시작: {self.experiment_id}")
        try:
            train_df, test_df, submission_df = self._load_data()
            train_df, test_df = self._add_cluster_ids(train_df, test_df)
            if selected_clusters is None:
                selected_clusters = list(self.config.cluster.mapping.keys())
            self._train_clusters(train_df, test_df, selected_clusters, predict=predict)
            submission_file = None
            if predict:
                submission_file = self._generate_submission(submission_df)
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
        """데이터 로딩"""
        self.logger.info("📊 데이터 로딩 중...")
        data_loader = DataLoader(self.config)
        return data_loader.load()
    
    def _add_cluster_ids(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """클러스터 ID 추가"""
        self.logger.info("🏷️ 클러스터 ID 매핑 중...")
        
        # branch_id → cluster_id 매핑 딕셔너리 생성
        branch_to_cluster = {}
        for cluster_id, branches in self.config.cluster.mapping.items():
            for branch in branches:
                branch_to_cluster[branch] = cluster_id
        
        train_df['cluster_id'] = train_df['branch_id'].map(branch_to_cluster)
        test_df['cluster_id'] = test_df['branch_id'].map(branch_to_cluster)
        
        # 매핑되지 않은 데이터 확인
        unmapped_train = train_df[train_df['cluster_id'].isna()]
        unmapped_test = test_df[test_df['cluster_id'].isna()]
        
        if len(unmapped_train) > 0 or len(unmapped_test) > 0:
            self.logger.warning(f"매핑되지 않은 데이터: train {len(unmapped_train)}, test {len(unmapped_test)}")
        
        return train_df, test_df
    
    def _train_clusters(self, train_df: pd.DataFrame, test_df: pd.DataFrame, selected_clusters: List[int], predict=False) -> None:
        self.logger.info(f"🎯 선택된 클러스터: {selected_clusters}")
        
        # 전체 valid 데이터 모으기용 리스트 ← 반드시 추가!
        all_valid_true = []
        all_valid_pred = []
        
        for cluster_id in selected_clusters:
            self.logger.info(f"{'='*20} 클러스터 {cluster_id} 처리 시작 {'='*20}")
            train_cluster = train_df[train_df['cluster_id'] == cluster_id].copy()
            test_cluster = test_df[test_df['cluster_id'] == cluster_id].copy()
            if len(train_cluster) == 0:
                self.logger.warning(f"클러스터 {cluster_id}에 학습 데이터가 없습니다. 건너뜁니다.")
                continue
            cluster_trainer = ClusterTrainer(self.config, cluster_id, self.experiment_id)
            result = cluster_trainer.train_and_predict(train_cluster, test_cluster)
            
            # predictions가 None일 수도 있으니 체크해서 저장
            if result['predictions'] is not None:
                self.cluster_results[cluster_id] = result['predictions']
            self.cluster_metrics[cluster_id] = result['metrics']
            self.logger.info(f"✅ 클러스터 {cluster_id} 완료")

            # (1) 클러스터별 베스트 모델(RMSE 기준) validation 예측값만 집계
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

        # (2) 전체 평가지표 계산 및 출력
        if all_valid_true and all_valid_pred:
            import numpy as np
            y_true_all = np.concatenate(all_valid_true)
            y_pred_all = np.concatenate(all_valid_pred)
            evaluator = ModelEvaluator(self.config)
            global_metrics = evaluator.evaluate(y_true_all, y_pred_all)
            self.logger.info("====== 전체(글로벌) 검증 평가지표 (클러스터별 베스트 모델 기준) ======")
            for metric, value in global_metrics.items():
                if value is not None:
                    self.logger.info(f"  {metric}: {value:.4f}")


    
    def _generate_submission(self, submission_df: pd.DataFrame) -> str:
        """제출 파일 생성"""
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
        submission_df = submission_df.merge(final_predictions[['id', self.config.data.target_column]], 
                                          on='id', how='left')
        
        # 파일 저장
        submission_file = self.file_manager.save_submission(submission_df, self.experiment_id)
        self.logger.info(f"📁 제출 파일 저장: {submission_file}")
        
        return submission_file
    
    def _save_results_and_report(self) -> None:
        """결과 저장 및 리포트 생성"""
        self.logger.info("💾 결과 저장 및 리포트 생성 중...")
        
        # 메트릭 저장
        self.file_manager.save_metrics(self.cluster_metrics, self.experiment_id)
        
        # 종합 리포트 생성
        evaluator = ModelEvaluator(self.config)
        evaluator.generate_report(self.cluster_metrics, self.experiment_id)

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="클러스터별 ML 파이프라인")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="설정 파일 경로")
    parser.add_argument("--clusters", nargs="*", type=int, 
                       help="학습할 클러스터 ID (기본값: 모든 클러스터)")
    parser.add_argument("--predict", action="store_true", help="최종 예측(제출)까지 실행")
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = ClusterMLPipeline(args.config)
    results = pipeline.run(selected_clusters=args.clusters, predict=args.predict)
    
    print(f"\n🎉 실험 완료! ID: {results['experiment_id']}")

if __name__ == "__main__":
    main()