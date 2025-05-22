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

# 클러스터 매핑 정의 (branch_id → cluster_id)
CLUSTER_MAP = {
    0: ['E', 'K', 'R', 'S'],
    1: ['F', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'Q'],
    2: ['A', 'B', 'C', 'D', 'G', 'H'],
}

def add_cluster_id(df):
    branch_to_cluster = {}
    for cluster_id, branches in CLUSTER_MAP.items():
        for b in branches:
            branch_to_cluster[b] = cluster_id
    df['cluster_id'] = df['branch_id'].map(branch_to_cluster)
    return df

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
        print(train_df.columns)

        # branch_id → cluster_id 매핑
        train_df = add_cluster_id(train_df)
        test_df = add_cluster_id(test_df)

        # 선택적 클러스터 학습 (예: --clusters 1 2)
        selected_clusters = getattr(args, 'clusters', list(CLUSTER_MAP.keys()))

        cluster_results = []
        cluster_metrics = []

        for cluster_id in selected_clusters:
            logger.info(f"===== [클러스터 {cluster_id}] 처리 시작 =====")
            train_c = train_df[train_df['cluster_id'] == cluster_id].copy()
            test_c = test_df[test_df['cluster_id'] == cluster_id].copy()

            # 데이터 분할
            x_train, x_valid, y_train, y_valid = data_split(args.split_type, train_c, target_column)
            X_test = test_c.copy()

            for df in [x_train, x_valid, X_test]:
                if 'branch_id' in df.columns:
                    df['branch_id'] = df['branch_id'].astype('category')

            # 파라미터 로드 및 병합
            default_params = load_params(args.model)
            user_params = parse_params(args.params)
            params = {**default_params, **user_params}
            
            params['enable_categorical'] = True

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
                params.update(study.best_trial.params)

            # 모델 학습
            logger.info("🚀 모델 학습 시작...")
            model = train_model(args.model, params, x_train, y_train, x_valid, y_valid)
            logger.info("✅ 학습 완료!")

            # 검증 예측 및 평가
            y_valid_pred = model.predict(x_valid)
            metrics = evaluate(y_valid, y_valid_pred)
            metrics['cluster'] = cluster_id
            cluster_metrics.append(metrics)

            logger.info("📊 검증 성능 지표:")
            for k, v in metrics.items():
                if k == 'cluster': continue
                if v is not None:
                    result = f"{k}: {v:.4f}"
                else:
                    result = f"{k}: 계산 불가"
                logger.info(result)

            # 결과 시각화
            if args.plot:
                logger.info("📈 결과 시각화 중...")
                plot_results(model, x_valid, y_valid, y_valid_pred)

            # 테스트 예측 및 결과 저장
            if args.submit:
                logger.info("📝 최종 모델 학습 및 제출 파일 생성 중...")

                # 전체 데이터로 재학습
                x_total = pd.concat([x_train, x_valid], axis=0)
                y_total = pd.concat([y_train, y_valid], axis=0)
                logger.info(f"전체 데이터 크기: {len(x_total)} 샘플")
                final_model = train_model(args.model, params, x_total, y_total)

                X_test = X_test.drop(columns=[target_column], errors="ignore")
                test_pred = final_model.predict(X_test)
                X_test['heat_demand'] = test_pred
                cluster_results.append(X_test[['id', 'heat_demand']])  # id 컬럼명은 실제 데이터에 맞게

        # 결과 통합 및 제출 파일 생성
        if args.submit and cluster_results:
            logger.info("클러스터별 결과 통합 및 제출 파일 생성 중...")
            final_submission = pd.concat(cluster_results).sort_values('id')
            submission_df = submission_df.drop(columns=['heat_demand'], errors="ignore")
            submission_df = submission_df.merge(final_submission, on='id', how='left')
            submission_df.to_csv(save_name, index=False, encoding='utf-8-sig')
            logger.info(f"📁 제출 파일 저장 완료: {save_name}")

        # 클러스터별 성능 저장
        save_metrics_to_csv(date_code, args.model, cluster_metrics)

        logger.info("프로그램 성공적으로 완료!")

    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
