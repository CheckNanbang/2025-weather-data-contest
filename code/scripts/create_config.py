# scripts/create_config.py
#!/usr/bin/env python3
"""
설정 파일 생성 스크립트
"""

import yaml
import argparse
from pathlib import Path

def create_default_config(output_path: str):
    """기본 설정 파일 생성"""
    
    config = {
        'experiment': {
            'name': 'heat_demand_prediction',
            'version': 'v1.0',
            'description': '클러스터별 열 수요 예측 실험'
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
            'type': 'time',  # 'random' 또는 'time'
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
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ 설정 파일 생성 완료: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="설정 파일 생성")
    parser.add_argument("--output", type=str, default="config/config.yaml",
                       help="출력 파일 경로 (기본값: config/config.yaml)")
    
    args = parser.parse_args()
    create_default_config(args.output)

if __name__ == "__main__":
    main()
