# src/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('cluster_ml')
        self.is_fitted = False
        
        # 전처리 도구들
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        self.branch_columns = []  # branch_id 더미 변수 컬럼명 저장
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 데이터에 대한 전처리 (fit + transform)"""
        self.logger.info("전처리 fit_transform 시작")
        df = df.replace(-99, np.nan)
        df_processed = df.copy()
        
        print("fit_transform")
        print(df.head())
        
        # 1. 날짜/시간 처리
        df_processed['tm_dt'] = pd.to_datetime(df_processed['tm'].astype(str), format="%Y%m%d%H")
        df_processed['month'] = df_processed['tm_dt'].dt.month
        
        # 2. branch_id 더미 변수 생성 (drop_first=True)
        branch_dummies = pd.get_dummies(df_processed['branch_id'], prefix='branch_id', drop_first=True)
        self.branch_columns = list(branch_dummies.columns)  # 컬럼명 저장
        df_processed = pd.concat([df_processed.drop('branch_id', axis=1), branch_dummies], axis=1)
        
        # 3. 불필요한 컬럼 제거
        df_processed = df_processed.drop(columns=['tm', 'tm_dt'])
        
        # 4. 타겟 컬럼 null값 제거 (학습 데이터만)
        if 'heat_demand' in df_processed.columns:
            df_processed = df_processed.dropna(subset=['heat_demand'])
        
        self.is_fitted = True
        self.logger.info(f"전처리 완료: {df_processed.shape}")
        return df_processed
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터에 대한 전처리 (transform only)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor가 fit되지 않았습니다.")
            
        self.logger.info("전처리 transform 시작")
        df = df.replace(-99, np.nan)
        df_processed = df.copy()
        
        # 1. 날짜/시간 처리
        df_processed['tm_dt'] = pd.to_datetime(df_processed['tm'].astype(str), format="%Y%m%d%H")
        df_processed['month'] = df_processed['tm_dt'].dt.month
        
        # 2. branch_id 더미 변수 생성 (학습 시와 동일한 컬럼 구조 유지)
        branch_dummies = pd.get_dummies(df_processed['branch_id'], prefix='branch_id', drop_first=True)
        
        # 학습 시 생성된 컬럼과 맞춤
        for col in self.branch_columns:
            if col not in branch_dummies.columns:
                branch_dummies[col] = 0  # 없는 컬럼은 0으로 추가
        
        # 순서도 맞춤
        branch_dummies = branch_dummies.reindex(columns=self.branch_columns, fill_value=0)
        
        df_processed = pd.concat([df_processed.drop('branch_id', axis=1), branch_dummies], axis=1)
        
        # 3. 불필요한 컬럼 제거
        df_processed = df_processed.drop(columns=['tm', 'tm_dt'])
        
        self.logger.info(f"전처리 변환 완료: {df_processed.shape}")
        return df_processed