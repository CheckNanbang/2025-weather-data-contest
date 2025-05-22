import os
import pandas as pd

def data_loader(dataset_name: str) -> tuple:
    data_path: str = "data"

    if dataset_name == "heat":
        target_column = "heat_demand"
        train_df = pd.read_csv(os.path.join(data_path, "train_heat.csv"), index_col=0)
        train_df.columns = train_df.columns.str.replace("train_heat.", "", regex=False)
        test_df = pd.read_csv(os.path.join(data_path, "test_heat.csv"), index_col=0)
        test_df.columns = test_df.columns.str.replace("train_heat.", "", regex=False)
        submission_df = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    else:
        raise ValueError(f"지원하지 않는 데이터셋 이름입니다: {dataset_name}")

    print(f"데이터 로드 완료: 학습 데이터 {train_df.shape}, 테스트 데이터 {test_df.shape}")
    return train_df, test_df, submission_df, target_column
