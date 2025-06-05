import yaml
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    version: str = "v1.0"
    description: str = ""

@dataclass
class DataConfig:
    dataset_name: str = "heat"
    target_column: str = "heat_demand"
    data_path: str = "data"
    prophet_target: str = 'y'

@dataclass
class ClusterConfig:
    mapping: Dict[int, List[str]] = field(default_factory=dict)

@dataclass
class SplitConfig:
    type: str = "time"
    test_size: float = 0.2
    random_state: int = 42
    time_split: Dict[str, int] = field(default_factory=lambda: {
        'holdout_start': 2023070101,
        'holdout_end': 2023123123
    })

@dataclass
class TrainingConfig:
    models: List[str] = field(default_factory=lambda: ["LGBM", "XGB"])
    tune_hyperparams: bool = True
    n_trials: int = 20
    early_stopping_rounds: int = 50
    prophet_params: Dict[str, Any] = field(default_factory=lambda: {
        'growth': 'linear',
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True
    })

@dataclass
class LoggingConfig:
    level: str = "INFO"
    save_plots: bool = True
    save_metrics: bool = True
    log_dir: str = "logs"
    plots_dir: str = "plots"
    results_dir: str = "results"

class Config:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self) -> None:
        """YAML 설정 파일 로드"""
        if not self.config_path.exists():
            self._create_default_config()
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 설정 객체 생성
        self.experiment = ExperimentConfig(**config_data.get('experiment', {}))
        self.data = DataConfig(**config_data.get('data', {}))
        self.cluster = ClusterConfig(**config_data.get('cluster', {}))
        self.split = SplitConfig(**config_data.get('split', {}))
        self.training = TrainingConfig(**config_data.get('training', {}))
        self.logging = LoggingConfig(**config_data.get('logging', {}))
    
    def _create_default_config(self) -> None:
        """기본 설정 파일 생성"""
        default_config = {
            'experiment': {
                'name': 'heat_demand_prediction',
                'version': 'v1.0',
                'description': '클러스터별 열 수요 예측'
            },
            'data': {
                'dataset_name': 'heat',
                'target_column': 'heat_demand',
                'data_path': 'data'
            },
            'cluster': {
                'mapping': {
                    0: ['E', 'K', 'R', 'S'],
                    1: ['F', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'Q'],
                    2: ['A', 'B', 'C', 'D', 'G', 'H']
                }
            },
            'split': {
                'type': 'time',
                'test_size': 0.2,
                'random_state': 42,
                'time_split': {
                    'holdout_start': 2023070101,
                    'holdout_end': 2023123123
                }
            },
            'training': {
                'models': ['LGBM', 'XGB'],
                'tune_hyperparams': True,
                'n_trials': 20,
                'early_stopping_rounds': 50
            },
            'logging': {
                'level': 'INFO',
                'save_plots': True,
                'save_metrics': True,
                'log_dir': 'logs',
                'plots_dir': 'plots',
                'results_dir': 'results'
            }
        }
        
        # 디렉토리 생성
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
