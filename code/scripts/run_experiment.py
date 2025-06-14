# scripts/run_experiment.py
#!/usr/bin/env python3
"""
실험 실행 스크립트
사용법:
    python scripts/run_experiment.py --config config/config.yaml --clusters 0 1
    python scripts/run_experiment.py --help
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import ClusterMLPipeline
import argparse

def main():
    parser = argparse.ArgumentParser(description="클러스터별 ML 파이프라인 실행")
    
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="설정 파일 경로 (기본값: config/config.yaml)")
    parser.add_argument("--clusters", nargs="*", type=int,
                       help="학습할 클러스터 ID 목록 (기본값: 모든 클러스터)")
    parser.add_argument("--models", nargs="+", help="학습할 모델명 리스트 (예: --models LGBM XGB)") 
    parser.add_argument("--dry-run", action="store_true",
                       help="실제 실행 없이 설정만 확인")
    parser.add_argument("--verbose", action="store_true",
                       help="상세 로그 출력")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 Dry-run 모드: 설정 확인만 수행")
        from src.utils.config import Config
        config = Config(args.config)
        print(f"✅ 설정 파일 로드 성공: {args.config}")
        print(f"📊 데이터셋: {config.data.dataset_name}")
        print(f"🎯 클러스터: {list(config.cluster.mapping.keys())}")
        print(f"🤖 모델: {config.training.models}")
        return
    
    try:
        pipeline = ClusterMLPipeline(args.config)
        
        # 여기서 덮어쓰기!
        if args.models:
            pipeline.config.training.models = args.models
        
        results = pipeline.run(selected_clusters=args.clusters)
        
        print("\n" + "="*50)
        print("🎉 실험 완료!")
        print(f"📋 실험 ID: {results['experiment_id']}")
        if results['submission_file']:
            print(f"📁 제출 파일: {results['submission_file']}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
