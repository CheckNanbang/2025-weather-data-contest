import os
import pandas as pd
from tqdm import tqdm
import argparse
import yaml
from datetime import datetime
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from log_utils import save_log
from models import xgboost

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost"], required=True)
    return parser.parse_args()

def save_predictions_to_csv(predictions, model_name, new_data=None):
    # 날짜 및 시간 정보 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 예측값만 저장
    df = pd.DataFrame({
        'Predicted': predictions,  # 예측값만
    })
    
    # 새 데이터가 제공되면, 예측값과 함께 그 데이터도 기록
    if new_data is not None:
        df['NewData'] = new_data  # 새로운 입력 데이터 추가 (실제값은 없지만)

    # logs 폴더에 저장
    logs_dir = 'logs'  # logs 폴더
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)  # 폴더가 없으면 생성
    
    # 파일 이름에 모델, 날짜, 시각 포함
    filename = os.path.join(logs_dir, f"predictions_{model_name}_{timestamp}.csv")
    
    # CSV로 저장
    df.to_csv(filename, index=False)
    print(f"[INFO] 예측 결과 저장됨: {filename}")


def main():
    args = parse_args()
    model_name = args.model

    print(f"[INFO] 모델 선택: {model_name}")

    # 하이퍼파라미터 로드
    print("[INFO] 하이퍼파라미터 불러오는 중...")
    with open(f"params/{model_name}_params.yaml") as f:
        params = yaml.safe_load(f)

    # 데이터 로드 (과거 데이터를 사용하여 모델 학습)
    print("[INFO] 데이터셋 로드 중...")
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # tqdm으로 학습 단계 진행률 표시
    print("[INFO] 모델 학습 중...")
    for _ in tqdm(range(1), desc="Training"):
        model, mse = xgboost.train(X_train, y_train, X_test, y_test, params)

    print(f"[INFO] 학습 완료. MSE: {round(mse, 4)}")

    # 로그 저장
    print("[INFO] 로그 저장 중...")
    metrics = {"mse": round(mse, 4)}
    save_log(params=params, metrics=metrics, model_name=model_name)

    # 새로운 예측용 데이터 (미래 데이터를 사용할 때)
    new_data = [[0.03, 0.25, -0.12, 0.02, 0.14, 0.11, -0.12, 0.04, -0.18, -0.08]]  # 예시 데이터
    print("[INFO] 새로운 데이터 예측 중...")

    # 예측
    preds = model.predict(new_data)

    # 예측 결과를 CSV 파일로 저장
    save_predictions_to_csv(preds, model_name, new_data=new_data)

    print("[INFO] 모든 작업 완료!")

if __name__ == "__main__":
    main()
