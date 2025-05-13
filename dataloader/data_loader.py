# data/data_loader.py
import os
import pandas as pd
import numpy as np

def data_loader(dataset_name: str) -> tuple:
    """
    주어진 데이터셋 이름에 따라 학습 및 테스트 데이터를 로드하는 함수.
    
    Args:
        dataset_name (str): 데이터셋 이름, 현재는 "heat"만 지원
        
    Returns:
        tuple: (학습 데이터프레임, 테스트 데이터프레임, 제출용 데이터프레임, 타겟 컬럼명)
    
    Raises:
        ValueError: 알 수 없는 데이터셋 이름이 제공될 경우
    """
    data_path: str = "data"

    if dataset_name == "heat":
        target_column = "train_heatheat_demand"
        train_df = pd.read_csv(os.path.join(data_path, "train_heat.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test_heat.csv"))
        submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    else:
        raise ValueError(f"지원하지 않는 데이터셋 이름입니다: {dataset_name}")
    
    # 컬럼명 정리 (특수문자 제거)
    train_df.columns = train_df.columns.str.replace(r'[^\w\s]', '', regex=True)
    test_df.columns = test_df.columns.str.replace(r'[^\w\s]', '', regex=True)

    # 결측값 처리 (먼저 전방 채우기 후 남은 결측값 -999로 채우기)
    train_df = train_df.ffill().fillna(-999)
    test_df = test_df.ffill().fillna(-999)
    
    print(f"데이터 로드 완료: 학습 데이터 {train_df.shape}, 테스트 데이터 {test_df.shape}")
    
    return train_df, test_df, submission_df, target_column