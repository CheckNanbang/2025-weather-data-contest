# src/data/data_loader.py
import os
import pandas as pd
from typing import Tuple
import logging

class DataLoader:
    """데이터 로딩 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('cluster_ml')
        
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 로드"""
        data_path = self.config.data.data_path
        dataset_name = self.config.data.dataset_name
        
        if dataset_name == "heat":
            train_file = os.path.join(data_path, "non_null_train_data.csv")
            test_file = os.path.join(data_path, "test_heat.csv")
            submission_file = os.path.join(data_path, "sample_submission.csv")
            
            train_df = pd.read_csv(train_file, index_col=0)
            test_df = pd.read_csv(test_file, index_col=0)
            train_df.columns = train_df.columns.str.replace("non_null_train_data.", "", regex=False)
            test_df.columns = test_df.columns.str.replace("non_null_train_data.", "", regex=False)
            submission_df = pd.read_csv(submission_file)
            
            self.logger.info(f"데이터 로드 완료: 학습 {train_df.shape}, 테스트 {test_df.shape}")
            
            return train_df, test_df, submission_df
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")