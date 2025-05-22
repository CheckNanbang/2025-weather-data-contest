# src/utils/file_manager.py
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class FileManager:
    """파일 저장 및 관리 클래스"""
    
    def __init__(self, config: Any):
        self.config = config
        self._create_directories()
    
    def _create_directories(self) -> None:
        """필요한 디렉토리 생성"""
        dirs = [
            self.config.logging.log_dir,
            self.config.logging.plots_dir, 
            self.config.logging.results_dir,
            "models",
            "submissions"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: Any, cluster_id: int, experiment_id: str, model_name: str) -> str:
        """모델 저장"""
        model_dir = Path("models") / experiment_id / f"cluster_{cluster_id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{model_name.lower()}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)
    
    def save_predictions(self, predictions: pd.DataFrame, cluster_id: int, 
                        experiment_id: str, model_name: str) -> str:
        """예측 결과 저장"""
        pred_dir = Path(self.config.logging.results_dir) / experiment_id / f"cluster_{cluster_id}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        pred_path = pred_dir / f"{model_name.lower()}_predictions.csv"
        predictions.to_csv(pred_path, index=False, encoding='utf-8-sig')
        
        return str(pred_path)
    
    def save_metrics(self, cluster_metrics: Dict[int, Dict[str, Any]], experiment_id: str) -> str:
        """메트릭 저장"""
        results_dir = Path(self.config.logging.results_dir) / experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON 형태로 저장
        metrics_path = results_dir / "cluster_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_metrics, f, indent=2, ensure_ascii=False, default=str)
        
        # CSV 형태로도 저장 (분석 용이성을 위해)
        metrics_list = []
        for cluster_id, metrics in cluster_metrics.items():
            for model_name, model_metrics in metrics.items():
                row = {'cluster_id': cluster_id, 'model': model_name}
                row.update(model_metrics)
                metrics_list.append(row)
        
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            csv_path = results_dir / "cluster_metrics.csv"
            metrics_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return str(metrics_path)
    
    def save_submission(self, submission_df: pd.DataFrame, experiment_id: str) -> str:
        """제출 파일 저장"""
        submission_dir = Path("submissions")
        submission_path = submission_dir / f"{experiment_id}_submission.csv"
        
        submission_df.to_csv(submission_path, index=False, encoding='utf-8-sig')
        return str(submission_path)
    
    def save_plots(self, plots: Dict[str, Any], cluster_id: int, 
                   experiment_id: str, model_name: str) -> Dict[str, str]:
        """플롯 저장"""
        if not self.config.logging.save_plots:
            return {}
            
        plot_dir = Path(self.config.logging.plots_dir) / experiment_id / f"cluster_{cluster_id}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        for plot_name, fig in plots.items():
            plot_path = plot_dir / f"{model_name.lower()}_{plot_name}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            saved_plots[plot_name] = str(plot_path)
        
        return saved_plots
