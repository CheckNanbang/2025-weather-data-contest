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
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 데이터에 대한 전처리 (fit + transform)"""
        self.logger.info("전처리 fit_transform 시작")
        df_processed = df.copy()
        
        # 1. 결측치 처리
        df_processed = self._handle_missing_values(df_processed, is_training=True)
        
        # 2. 파생 변수 생성
        # df_processed = self._create_features(df_processed)
        
        # 3. 범주형 변수 인코딩
        df_processed = self._encode_categorical(df_processed, is_training=True)
        
        # 4. 수치형 변수 스케일링
        # df_processed = self._scale_numerical(df_processed, is_training=True)
        
        # 5. 시계열 특성 추가
        # df_processed = self._add_time_features(df_processed)
        
        self.is_fitted = True
        self.logger.info(f"전처리 완료: {df_processed.shape}")
        return df_processed
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터에 대한 전처리 (transform only)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor가 fit되지 않았습니다.")
            
        self.logger.info("전처리 transform 시작")
        df_processed = df.copy()
        
        # 동일한 전처리 과정 (is_training=False)
        df_processed = self._handle_missing_values(df_processed, is_training=False)
        # df_processed = self._create_features(df_processed)
        df_processed = self._encode_categorical(df_processed, is_training=False)
        # df_processed = self._scale_numerical(df_processed, is_training=False)
        # df_processed = self._add_time_features(df_processed)
        
        self.logger.info(f"전처리 변환 완료: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """결측치 처리"""
        # 수치형 변수: 평균/중앙값으로 대체
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != self.config.data.target_column]
        
        if is_training:
            for col in numerical_cols:
                if df[col].isnull().sum() > 0:
                    fill_value = df[col].median()
                    self.feature_stats[f'{col}_fill'] = fill_value
                    df[col].fillna(fill_value, inplace=True)
        else:
            for col in numerical_cols:
                if f'{col}_fill' in self.feature_stats:
                    df[col].fillna(self.feature_stats[f'{col}_fill'], inplace=True)
        
        # 범주형 변수: 최빈값으로 대체
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if is_training:
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                    self.feature_stats[f'{col}_fill'] = fill_value
                    df[col].fillna(fill_value, inplace=True)
        else:
            for col in categorical_cols:
                if f'{col}_fill' in self.feature_stats:
                    df[col].fillna(self.feature_stats[f'{col}_fill'], inplace=True)
        
        return df
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        # 시간 관련 파생 변수 (train_heattm 컬럼이 있다고 가정)
        if 'train_heattm' in df.columns:
            df['hour'] = df['train_heattm'] % 100
            df['day'] = (df['train_heattm'] // 100) % 100
            df['month'] = (df['train_heattm'] // 10000) % 100
            df['year'] = df['train_heattm'] // 1000000
            
            # 시간대 구분
            df['time_period'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['night', 'morning', 'afternoon', 'evening'])
            
            # 요일 추가 (간단한 예시)
            df['weekday'] = df['day'] % 7
            
            # 계절 구분
            df['season'] = df['month'].map(lambda x: 
                'spring' if x in [3,4,5] else
                'summer' if x in [6,7,8] else  
                'autumn' if x in [9,10,11] else 'winter')
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col == self.config.data.target_column:
                continue
                
            if is_training:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder
            else:
                if col in self.encoders:
                    encoder = self.encoders[col]
                    # 새로운 카테고리는 -1로 처리
                    df[col] = df[col].map(lambda x: encoder.transform([str(x)])[0] 
                                        if str(x) in encoder.classes_ else -1)
        
        return df
    
    def _scale_numerical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """수치형 변수 스케일링"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        scale_cols = [col for col in numerical_cols 
                     if col not in ['id', self.config.data.target_column, 'cluster_id']]
        
        if is_training:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                scaler = self.scalers['numerical']
                df[scale_cols] = scaler.transform(df[scale_cols])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 특성 추가"""
        # 시간 기반 순환 특성 (sin, cos 변환)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df