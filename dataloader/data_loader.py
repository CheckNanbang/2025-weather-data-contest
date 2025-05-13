import os
import pandas as pd

def data_loader(dataset_name: str) -> pd.DataFrame:
    """
    주어진 데이터셋 이름에 따라 학습 및 테스트 데이터를 로드하는 함수.
    """
    data_path: str = "data"

    if dataset_name == "heat":
        target_column = "train_heatheat_demand"
        train_df = pd.read_csv(os.path.join(data_path, "train_heat.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test_heat.csv"))
        submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    # 컬럼명 정리
    train_df.columns = train_df.columns.str.replace(r'[^\w\s]', '', regex=True)
    test_df.columns = test_df.columns.str.replace(r'[^\w\s]', '', regex=True)

    # 결측값 처리
    train_df = train_df.ffill().fillna(-999)
    test_df = test_df.ffill().fillna(-999)
    
    return train_df, test_df, submission_df, target_column
