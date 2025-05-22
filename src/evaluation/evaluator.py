# src/evaluation/evaluator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import logging
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score
)

class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('cluster_ml')
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """회귀 모델 평가 지표 계산"""
        metrics = {}
        
        try:
            metrics['MAE'] = mean_absolute_error(y_true, y_pred)
            metrics['MSE'] = mean_squared_error(y_true, y_pred)
            metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['R2'] = r2_score(y_true, y_pred)
            metrics['EVS'] = explained_variance_score(y_true, y_pred)
            
            # MAPE (음수 값이 있을 경우 처리)
            if np.all(y_true > 0):
                metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
            else:
                metrics['MAPE'] = None
                
        except Exception as e:
            self.logger.warning(f"평가 지표 계산 중 오류: {e}")
            
        return metrics
    
    def create_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                    model: Any = None) -> Dict[str, plt.Figure]:
        """평가 시각화 생성"""
        plots = {}
        
        # 1. 실제값 vs 예측값 산점도
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('True Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title('True vs Predicted Values', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # R² 점수 표시
        r2 = r2_score(y_true, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plots['scatter'] = fig1
        
        # 2. 잔차 분포
        residuals = y_true - y_pred
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 잔차 히스토그램
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residuals Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 실제값 vs 잔차
        ax3.scatter(y_true, residuals, alpha=0.6)
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('True Values', fontsize=12)
        ax3.set_ylabel('Residuals', fontsize=12)
        ax3.set_title('Residuals vs True Values', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        plots['residuals'] = fig2
        
        # 3. 특성 중요도 (모델이 제공되는 경우)
        if model is not None:
            try:
                importance = model.get_feature_importance()
                if importance:
                    # 상위 20개 특성만 표시
                    top_features = dict(list(importance.items())[:20])
                    
                    fig3, ax4 = plt.subplots(figsize=(12, 8))
                    features = list(top_features.keys())
                    values = list(top_features.values())
                    
                    bars = ax4.barh(features, values)
                    ax4.set_xlabel('Importance', fontsize=12)
                    ax4.set_ylabel('Features', fontsize=12)
                    ax4.set_title('Top 20 Feature Importance', fontsize=14)
                    ax4.grid(True, alpha=0.3, axis='x')
                    
                    # 색상 그라데이션
                    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    plt.tight_layout()
                    plots['importance'] = fig3
                    
            except Exception as e:
                self.logger.warning(f"특성 중요도 시각화 실패: {e}")
        
        return plots
    
    def generate_report(self, cluster_metrics: Dict[int, Dict[str, Any]], 
                       experiment_id: str, global_metrics: Dict[str, float] = None) -> str:
        """종합 리포트 생성"""
        self.logger.info("종합 리포트 생성 중...")
        
        report_lines = []
        report_lines.append(f"# 실험 리포트: {experiment_id}")
        report_lines.append(f"생성 시간: {pd.Timestamp.now()}")
        report_lines.append("")
        
        # 클러스터별 성능 요약
        report_lines.append("## 클러스터별 성능 요약")
        report_lines.append("")
        
        summary_data = []
        for cluster_id, models_metrics in cluster_metrics.items():
            for model_name, metrics in models_metrics.items():
                summary_data.append({
                    'Cluster': cluster_id,
                    'Model': model_name,
                    'RMSE': metrics.get('RMSE', 'N/A'),
                    'MAE': metrics.get('MAE', 'N/A'),
                    'R2': metrics.get('R2', 'N/A')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            report_lines.append(summary_df.to_string(index=False))
            report_lines.append("")
            
            # 최고 성능 모델
            if 'RMSE' in summary_df.columns:
                numeric_rmse = pd.to_numeric(summary_df['RMSE'], errors='coerce')
                best_idx = numeric_rmse.idxmin()
                best_model = summary_df.loc[best_idx]
                
                report_lines.append("## 최고 성능 모델")
                report_lines.append(f"- 클러스터: {best_model['Cluster']}")
                report_lines.append(f"- 모델: {best_model['Model']}")
                report_lines.append(f"- RMSE: {best_model['RMSE']}")
                report_lines.append(f"- R²: {best_model['R2']}")
                report_lines.append("")
        
        # 클러스터별 상세 분석
        report_lines.append("## 클러스터별 상세 분석")
        report_lines.append("")
        
        for cluster_id in sorted(cluster_metrics.keys()):
            report_lines.append(f"### 클러스터 {cluster_id}")
            
            cluster_models = cluster_metrics[cluster_id]
            for model_name, metrics in cluster_models.items():
                report_lines.append(f"#### {model_name}")
                for metric, value in metrics.items():
                    if value is not None:
                        if isinstance(value, float):
                            report_lines.append(f"- {metric}: {value:.4f}")
                        else:
                            report_lines.append(f"- {metric}: {value}")
                report_lines.append("")
        
        if global_metrics:
            report_lines.append("## 전체(글로벌) 검증 평가지표 (클러스터별 베스트 모델 기준)")
            for metric, value in global_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        report_lines.append(f"- {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"- {metric}: {value}")
            report_lines.append("")
        
        # 리포트 파일 저장
        from pathlib import Path
        report_dir = Path(self.config.logging.results_dir) / experiment_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / "experiment_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"리포트 저장 완료: {report_path}")
        return str(report_path)