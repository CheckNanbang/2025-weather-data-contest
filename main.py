import os
import sys

import pandas as pd
from datetime import datetime
import optuna

# 로컬 모듈 임포트
from dataloader.data_loader import data_loader
from utils.data_split import data_split
from utils.load_params import load_params
from models.train_model import train_model
from utils.evaluation import evaluate
from utils.cli_utils import parse_args, parse_params
from utils.log_utils import setup_logger, save_metrics_to_csv
from utils.train_utils import objective, plot_results

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def main():
    """
    메인 실행 함수
    """
    # 명령줄 인자 파싱
    args = parse_args()
    now = datetime.now()
    date_code = now.strftime("%m%d%H%M%S")
    save_name = f"{args.model}_{args.dataset}_{args.split_type}_{date_code}.csv"

    log_filename = f"{args.model}_{date_code}.log"
    log_path = os.path.join("logs", log_filename)
    os.makedirs("logs", exist_ok=True)
    logfile = open(log_path, "a", encoding="utf-8-sig")
    sys.stdout = Tee(sys.stdout, logfile)
    sys.stderr = Tee(sys.stderr, logfile)
    logger = setup_logger(log_filename)
    logger.info(f"프로그램 시작: {args.model} 모델, {args.dataset} 데이터셋, {args.split_type} 분할 방식")
    
    try:
        # 데이터 로딩
        logger.info("데이터 로딩 중...")
        train_df, test_df, submission_df, target_column = data_loader(args.dataset)
        
        # 데이터 분할
        logger.info(f"데이터 {args.split_type} 방식으로 분할 중...")
        x_train, x_valid, y_train, y_valid = data_split(args.split_type, train_df, target_column)
        
        # 테스트 데이터 준비
        X_test = test_df.copy()
        
        # 불필요한 컬럼 제거
        drop_cols = ["train_heatbranch_id"]
        for col in drop_cols:
            if col in x_train.columns:
                x_train = x_train.drop(columns=[col])
                logger.info(f"학습 데이터에서 {col} 컬럼 제거")
            if col in x_valid.columns:
                x_valid = x_valid.drop(columns=[col])
                logger.info(f"검증 데이터에서 {col} 컬럼 제거")
            if col in X_test.columns:
                X_test = X_test.drop(columns=[col])
                logger.info(f"테스트 데이터에서 {col} 컬럼 제거")
        
        # 기본 파라미터 로드
        default_params = load_params(args.model)
        
        # 사용자 지정 파라미터 적용
        user_params = parse_params(args.params)
        params = {**default_params, **user_params}
        
        logger.info("🔧 최종 사용 파라미터:")
        for k, v in params.items():
            logger.info(f"  - {k}: {v}")
        
        # Optuna 하이퍼파라미터 튜닝
        if args.tune:
            logger.info(f"Optuna 하이퍼파라미터 튜닝 시작 (시도 횟수: {args.n_trials})")
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, args.model, x_train, y_train, x_valid, y_valid), 
                n_trials=args.n_trials
            )
            
            logger.info("🎯 최적 파라미터:")
            for k, v in study.best_trial.params.items():
                logger.info(f"  - {k}: {v}")
            
            # 최적 파라미터 적용
            params.update(study.best_trial.params)
        
        # 모델 학습
        logger.info("🚀 모델 학습 시작...")
        model = train_model(args.model, params, x_train, y_train, x_valid, y_valid)
        logger.info("✅ 학습 완료!")
        
        # 예측 및 평가
        y_valid_pred = model.predict(x_valid)
        metrics = evaluate(y_valid, y_valid_pred)
        
        logger.info("📊 검증 성능 지표:")
        for k, v in metrics.items():
            if v is not None:
                result = f"{k}: {v:.4f}"
            else:
                result = f"{k}: 계산 불가"
            logger.info(result)
        
        save_metrics_to_csv(date_code, args.model, metrics)
        
        # 결과 시각화
        if args.plot:
            logger.info("📈 결과 시각화 중...")
            plot_results(model, x_valid, y_valid, y_valid_pred)
        
        # 최종 예측 및 제출 파일 생성
        if args.submit:
            logger.info("📝 최종 모델 학습 및 제출 파일 생성 중...")
            
            # 전체 데이터로 모델 재학습
            x_total = pd.concat([x_train, x_valid], axis=0)
            y_total = pd.concat([y_train, y_valid], axis=0)
            
            logger.info(f"전체 데이터 크기: {len(x_total)} 샘플")
            final_model = train_model(args.model, params, x_total, y_total)
            
            # 테스트 데이터 타겟 컬럼 제거 (있는 경우)
            X_test = X_test.drop(columns=[target_column], errors="ignore")
            
            # 예측 수행
            test_pred = final_model.predict(X_test)
            
            # 제출 파일 저장
            submission_df['heat_demand'] = test_pred
            submission_df.to_csv(save_name, index=False, encoding='utf-8-sig')
            
            logger.info(f"📁 제출 파일 저장 완료: {save_name}")
        
        logger.info("프로그램 성공적으로 완료!")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()