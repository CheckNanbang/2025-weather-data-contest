import pandas as pd
import numpy as np
import logging
from typing import Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm

class DataPreprocessor:
    """데이터 전처리 클래스"""

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger('cluster_ml')
        self.is_fitted = False
        self.scalers = {}
        self.encoders = {}

    # ========================
    # fit_transform / transform
    # ========================
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 데이터 전처리 (fit + transform)"""
        self.logger.info("🔨 전처리 fit_transform 시작")
        
        # 데이터 검증
        if df.empty:
            raise ValueError("입력 데이터프레임이 비어있습니다.")
            
        self.logger.info(f"입력 데이터 크기: {df.shape}")
        self.logger.info(f"입력 데이터 컬럼: {df.columns.tolist()}")
        
        # prophet 전용    
        if 'Prophet' in self.config.training.models:
            df = df.rename(columns={
                self.config.data.target_column: 'y',
                'tm': 'ds'
            })
        
        # 기본 전처리
        df = self._base_preprocess(df)
        # df = df.dropna(subset=['heat_demand'])
        
        # 파생 변수 추가
        df = self._create_features(df)
        
        # 수치형 변수 스케일링
        df = self._scale_numerical(df, is_training=True)
        
        # 범주형 변수 인코딩
        df = self._encode_categorical(df, is_training=True)
        
        # 시계열 변수 추가
        df = self._add_time_features(df)
        
        # tm 컬럼 제거 (필요시)
        df = df.drop(columns=['tm'])
        self.is_fitted = True
        self.logger.info(f"전처리 완료: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터 전처리 (transform only)"""
        self.logger.info("🔨 테스트 데이터 전처리 transform 시작")
        if not self.is_fitted:
            raise ValueError("Preprocessor가 fit되지 않았습니다.")
        self.logger.info(f"전처리 transform 시작: shape={df.shape}")
        # 기본 전처리
        df = self._base_preprocess(df)
        # df = df.dropna(subset=['heat_demand'])
        print(df.shape)
        # 파생 변수 추가
        df = self._create_features(df)
        print(df.shape)
        # 수치형 변수 스케일링
        df = self._scale_numerical(df, is_training=True)
        print(df.shape)
        # 범주형 변수 인코딩
        df = self._encode_categorical(df, is_training=True)
        print(df.shape)
        # 시계열 변수 추가
        df = self._add_time_features(df)
        # tm 컬럼 제거 (필요시)
        df = df.drop(columns=['tm'])
        self.logger.info(f"전처리 변환 완료: {df.shape}")
        return df

    # ========================
    # 기본 전처리
    # ========================
    def _base_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리: 날짜 변환, 더미 생성, 불필요 컬럼 제거"""
        self.logger.info("🔨 기본 전처리")
        df = df.copy()
        
        # 데이트 타임 변경 & 월 추가
        df['tm'] = pd.to_datetime(df['tm'].astype(str), format="%Y%m%d%H")
        df['month'] = df['tm'].dt.month
        
        # 263069개의 결측치 -> na로 대체
        df = df.replace(-99, np.nan)

        # 풍향 -9.9 값을 NaN으로 변경
        df['wd'] = df['wd'].replace(-9.9, np.nan)
        
        # branch_dummies = pd.get_dummies(df['branch_id'], prefix='branch_id', drop_first=True)
        # df = pd.concat([df.drop('branch_id', axis=1), branch_dummies], axis=1)
        # df = df.drop(columns=['tm', 'tm_dt'])
        return df

    # ========================
    # 결측치 처리
    # ========================
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """결측치 처리: 패턴별 si 채움, 장단기 결측치 보간"""
        self.logger.info("🔨 결측치 처리")
        # ... (패턴 그룹 및 결측치 처리 로직 동일)
        return df

    # ========================
    # 파생 변수 생성
    # ========================
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성: 시간, 계절, 풍향/풍속, 이슬점 등"""
        self.logger.info("🔨 파생 변수 생성")
        
        if 'Prophet' in self.config.training.models:
            # Prophet 사용 시 특징 생성 생략
            return df[['ds', 'y', 'branch_id']] 
        
        else:
            df['year'] = df['tm'].dt.year
            df['day'] = df['tm'].dt.day
            df['hour'] = df['tm'].dt.hour

            # # 시간대 구분
            # df['time_period'] = pd.cut(df['hour'], 
            #                         bins=[0, 6, 12, 18, 24], 
            #                         labels=['night', 'morning', 'afternoon', 'evening'])
                
            # 요일 추가 (간단한 예시)
            df['weekday'] = df['day'] % 7
                
            # 계절 구분
            df['season'] = df['month'].map(lambda x: 
                'spring' if x in [3,4,5] else
                'summer' if x in [6,7,8] else  
                'autumn' if x in [9,10,11] else 'winter')
            
            #풍향그룹
            self.logger.info("🔨 풍향")
            df['wd_group'] = df['wd'].apply(self._categorize_wind_direction)

            #풍속그룹
            self.logger.info("🔨 풍속")
            df['ws_group'] = df['ws'].apply(self._categorize_ws)

            #이슬점
            self.logger.info("🔨 이슬점 계산")
            df['dew_point'] = df.apply(lambda row: self._dew_point(row['ta'], row['hm']), axis=1)
            
            # lag 변수 추가
            df = self._add_lag_features(df, lag=1)

            #이동평균
            target_cols = ['wd', 'ws', 'rn_hr1', 'rn_day']
            df = self._add_moving_averages(df, target_cols, window_size=3)
            
            # #trend ,계절성변수
            # daily_avg = (
            # df.copy()
            # .assign(date=lambda x: x['tm'].dt.date)
            # .groupby('date')['heat_demand']
            # .mean()
            # )

            # # STL 분해
            # result = sm.tsa.seasonal_decompose(daily_avg, model='additive', period=365)
            # df['trend'] = result.trend
            # df['seasonal'] =result.seasonal
            # df['residual'] = result.resid
            
        return df

    # ========================
    # 범주형 인코딩
    # ========================
    def _encode_categorical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        self.logger.info("🔨 범주형 변수 인코딩")
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
                    df[col] = df[col].map(lambda x: encoder.transform([str(x)])[0] 
                                          if str(x) in encoder.classes_ else -1)
        return df

    # ========================
    # 수치형 스케일링
    # ========================
    def _scale_numerical(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """수치형 변수 스케일링"""
        self.logger.info("🔨 수치형 변수 스케일링")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        # exclude_cols = ['id', 'cluster_id', 'heat_demand', 'log_heat_demand']
        # scale_cols = [col for col in numerical_cols if col not in exclude_cols]
        # if is_training:
        #     scaler = StandardScaler()
        #     df[scale_cols] = scaler.fit_transform(df[scale_cols])
        #     self.scalers['numerical'] = scaler
        # else:
        #     if 'numerical' in self.scalers:
        #         scaler = self.scalers['numerical']
        #         df[scale_cols] = scaler.transform(df[scale_cols])
        return df

    # ========================
    # 시계열 특성 추가 (퓨리에/순환)
    # ========================
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시계열 특성 추가: 퓨리에, 순환(sin/cos) 변환 등"""
        self.logger.info("🔨 시계열 특성 추가")
        df_copy = df.copy()

        df_copy = df_copy.set_index('tm')              # tm을 인덱스로 설정
        
        # 일 평균 시계열 (STL용)
        daily_avg = df_copy['ta'].resample('D').mean()

        df_copy = df_copy.reset_index()  # tm을 다시 칼럼으로

        # STL 분해
        # period = 30  # 월간 계절성 가정

        # result = sm.tsa.seasonal_decompose(daily_avg, model='additive', period=period)

        # # STL 분해 후: NaN 포함된 resid
        # resid = result.resid

        # # 선형 보간 + 앞뒤 결측 보완
        # resid_filled = resid.interpolate(method='time').ffill().bfill()  # 시간기반 보간 + 앞뒤 결측 처리

        # df = self._process_fft_filtering(resid_filled, df)

        """시계열 특성 추가"""
        # 시간 기반 순환 특성 (sin, cos 변환)
        df['hour_of_week'] = df['tm'].dt.dayofweek * 24 + df['hour']
        df['day_of_month'] = df['tm'].dt.day

        # 하루 단위 
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        # 일주일 단위
        df['weekhour_sin'] = np.sin(2 * np.pi * df['hour_of_week'] / 168)
        df['weekhour_cos'] = np.cos(2 * np.pi * df['hour_of_week'] / 168)
        # 월 단위 
        df['month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
        df['month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)

        return df

    # ========================
    # 기타 유틸 함수
    # ========================
    def _categorize_wind_direction(self, degree):
        """풍향을 카테고리로 변환"""
        
        if 45 <= degree < 135:
            return 'E'
        elif 135 <= degree < 225:
            return 'S'
        elif 225 <= degree < 315:
            return 'W'
        else:
            return 'N'

    def _categorize_ws(self, ws):
        """풍속을 카테고리로 변환"""
        if ws < 1:
            return '정지'
        elif ws < 4:
            return '약풍'
        elif ws < 9:
            return '중풍'
        else:
            return '강풍'

    def _dew_point(self, temp, humidity):
        """이슬점 계산"""
        a, b = 17.27, 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100)
        return (b * alpha) / (a - alpha)

    def _add_lag_features(self, df, lag=1)-> pd.DataFrame:
        '''수치형 변수에 lag 변수 추가'''    
        # 사용할 변수와 lag 시기 정의
        target_vars = ['ta', 'hm', 'si', 'ta_chi']
        lags = [1, 2, 24, 25]

        # 각 변수별, 각 lag별로 shift 적용
        for var in target_vars:
            for lag in lags:
                df[f'{var}_lag_{lag}'] = df.groupby('branch_id')[var].shift(lag)

        return df
    
    def _add_moving_averages(self, df, target_cols, window_size=3) -> pd.DataFrame:
        '''branch_id별 이동평균 추가'''

        if 'branch_id' not in df.columns or 'tm' not in df.columns:
         raise ValueError("⚠️ 'branch_id' 또는 'tm' 컬럼이 누락되었습니다.")
    
        #   시간 정렬 (branch별 rolling을 적용하려면 반드시 필요)
        df = df.sort_values(['branch_id', 'tm'])

        for col in target_cols:
            if col in df.columns:
                df[f'{col}_ma'] = (
                 df.groupby('branch_id')[col]
                      .transform(lambda x: x.rolling(window=window_size, min_periods=window_size).mean())
                )
                print(f"✅ {col} 이동평균 계산 완료")
            else:
                print(f"❌ {col} 컬럼 없음")
        return df