# src/data/preprocessor.py
import pandas as pd
import numpy as np
from typing import Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
        df_processed['tm_dt'] = pd.to_datetime(df_processed['tm'].astype(str), format="%Y%m%d%H")
        df_processed['month'] = df_processed['tm_dt'].dt.month
        branch_dummies = pd.get_dummies(df_processed['branch_id'], prefix='branch_id', drop_first=True)
        df_processed = pd.concat([df_processed.drop('branch_id', axis=1), branch_dummies], axis=1)
        df_processed = df_processed.drop(columns=['tm', 'tm_dt'])

        # df_processed = df.iloc[:,:]
        
        # # 0.기본전처리
        # df_processed = self._basic_process(df_processed, is_training=True)
        

        # # 1. 타겟 칼럼 null값 삭제
        # df_processed = df_processed.dropna(subset=['heat_demand'])

        # # 2. 타겟 로그변환
        # df['log_heat_demand'] = np.log1p(df['heat_demand'])
        
        # # 3. 결측치 처리
        # df_processed = self._handle_missing_values(df_processed, is_training=True)
        
        # # 4. 파생 변수 생성
        # df_processed = self._create_features(df_processed)
    
        # # 5. 수치형 변수 스케일링
        # df_processed = self._scale_numerical(df_processed, is_training=True)
        
        # # 6. 범주형 변수 인코딩
        # df_processed = self._encode_categorical(df_processed, is_training=True)
                
        # # 7. 시계열 특성 추가 cos,sin + 퓨리에 
        # df_processed = self._add_time_features(df_processed)

        # # 8. 시계열 컬럼으로 인해 생긴 null값 제거
        # df_processed = df_processed.dropna()

        self.is_fitted = True
        self.logger.info(f"전처리 완료: {df_processed.shape}")
        return df_processed
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """테스트 데이터에 대한 전처리 (transform only)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor가 fit되지 않았습니다.")
            
        self.logger.info("전처리 transform 시작")
        df_processed = df.copy()

        # 임시 테스트
        print("데이터 shape:", df_processed.shape)
        df_processed['tm_dt'] = pd.to_datetime(df_processed['tm'].astype(str), format="%Y%m%d%H")
        df_processed['month'] = df_processed['tm_dt'].dt.month
        branch_dummies = pd.get_dummies(df_processed['branch_id'], prefix='branch_id', drop_first=True)
        df_processed = pd.concat([df_processed.drop('branch_id', axis=1), branch_dummies], axis=1)
        df_processed = df_processed.drop(columns=['tm', 'tm_dt'])
        
        # 동일한 전처리 과정 (is_training=False)
        # df_processed = self._basic_process(df_processed, is_training=False)
        # df_processed = self._handle_missing_values(df_processed, is_training=False)
        # df_processed = self._create_features(df_processed)
        # df_processed = self._scale_numerical(df_processed, is_training=False)
        # df_processed = self._encode_categorical(df_processed, is_training=False)
        # df_processed = self._add_time_features(df_processed)
        # df_processed = df_processed.dropna(subset=['heat_demand'])
        
        self.logger.info(f"전처리 변환 완료: {df_processed.shape}")
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """결측치 처리"""
        # 1. si변수
        # A/B/C/F용 패턴
        pattern_group_1 = [
            ((1, 1), (2, 6), 19, 7),
            ((2, 6), (3, 4), 20, 7),
            ((3, 4), (4, 7), 20, 6),
            ((4, 7), (4, 13), 21, 6),
            ((4, 13), (8, 28), 21, 5),
            ((8, 28), (9, 3), 21, 6),
            ((9, 3), (10, 13), 20, 6),
            ((10, 13), (11, 3), 19, 6),
            ((11, 3), (12, 31), 19, 7),
        ]

        # D/E/G/H/I/J/K/S용 패턴
        pattern_group_2 = [
            ((1, 1), (2, 5), 19, 7),
            ((2, 5), (3, 3), 20, 7),
            ((3, 3), (4, 7), 20, 6),
            ((4, 7), (4, 13), 21, 6),
            ((4, 13), (8, 28), 21, 5),
            ((8, 28), (9, 3), 21, 6),
            ((9, 3), (10, 13), 20, 6),
            ((10, 13), (11, 4), 19, 6),
            ((11, 4), (12, 31), 19, 7),
        ]

        #L,M,N (유사)
        pattern_group_3 = [
            ((1, 1),   (2, 10), 19, 7),  # 1월 1일 ~ 2월 10일 07시까지
            ((2, 10),  (2, 25), 20, 7),  # 2월 10일 ~ 2월 25일 07시까지
            ((2, 25),  (4, 9),  20, 6),  # 2월 25일 ~ 4월 9일 06시까지
            ((4, 9),   (4, 19), 20, 5),  # 4월 9일 ~ 4월 19일 05시까지
            ((4, 19),  (8, 26), 21, 5),  # 4월 19일 ~ 8월 26일 05시까지
            ((8, 26),  (9, 4),  20, 5),  # 8월 26일 ~ 9월 4일 05시까지
            ((9, 4),   (10, 7), 20, 6),  # 9월 4일 ~ 10월 7일 06시까지
            ((10, 7),  (11, 13), 19, 6), # 10월 7일 ~ 11월 13일 06시까지
            ((11, 13), (12, 31), 19, 7), # 11월 13일 ~ 12월 31일 07시까지
        ]

        #R
        pattern_group_R = [
            ((1, 1),   (1, 31), 19, 7),  # 1월 1일 ~ 1월 31일 07시까지
            ((1, 31),  (3, 2),  20, 7),  # 1월 31일 ~ 3월 2일 07시까지
            ((3, 2),   (4, 9),  20, 6),  # 3월 2일 ~ 4월 9일 06시까지
            ((4, 9),   (4, 15), 21, 6),  # 4월 9일 ~ 4월 15일 06시까지
            ((4, 15),  (8, 26), 21, 5),  # 4월 15일 ~ 8월 26일 05시까지
            ((8, 26),  (9, 2),  21, 6),  # 8월 26일 ~ 9월 2일 06시까지
            ((9, 2),   (10, 14), 20, 6), # 9월 2일 ~ 10월 14일 06시까지
            ((10, 14), (11, 8), 19, 6),  # 10월 14일 ~ 11월 8일 06시까지
            ((11, 8),  (2, 2),  19, 7),  # 11월 8일 ~ 다음해 2월 2일 07시까지
        ]

        #O
        pattern_group_O = [
            ((1, 1),   (2, 6),  19, 7),  # 1월 1일 ~ 2월 6일 07시까지
            ((2, 6),   (3, 2),  20, 7),  # 2월 6일 ~ 3월 2일 07시까지
            ((3, 2),   (4, 11), 20, 6),  # 3월 2일 ~ 4월 11일 06시까지
            ((4, 11),  (4, 13), 21, 6),  # 4월 11일 ~ 4월 13일 06시까지
            ((4, 13),  (8, 30), 21, 5),  # 4월 13일 ~ 8월 30일 05시까지
            ((8, 30),  (9, 1),  21, 6),  # 8월 30일 ~ 9월 1일 06시까지
            ((9, 1),   (10, 11), 20, 6), # 9월 1일 ~ 10월 11일 06시까지
            ((10, 11), (11, 7), 19, 6),  # 10월 11일 ~ 11월 7일 06시까지
            ((11, 7),  (12, 31), 19, 7), # 11월 7일 ~ 12월 31일까지
        ]

        #P
        pattern_group_P = [
            ((1, 20),  (2, 6),  19, 7),  # 1월 20일 ~ 2월 6일 07시까지
            ((2, 6),   (3, 3),  20, 7),  # 2월 6일 ~ 3월 3일 07시까지
            ((3, 3),   (4, 10), 20, 6),  # 3월 3일 ~ 4월 10일 06시까지
            ((4, 10),  (4, 13), 21, 6),  # 4월 10일 ~ 4월 13일 06시까지
            ((4, 13),  (8, 30), 21, 5),  # 4월 13일 ~ 8월 30일 05시까지
            ((8, 30),  (9, 2),  21, 6),  # 8월 30일 ~ 9월 2일 06시까지
            ((9, 2),   (10, 12), 20, 6), # 9월 2일 ~ 10월 12일 06시까지
            ((10, 12), (11, 6), 19, 6),  # 10월 12일 ~ 11월 6일 06시까지
            ((11, 6),  (12, 31), 19, 7), # 11월 6일 ~ 12월 31일까지
        ]

        #Q
        pattern_group_Q = [
            ((1, 1), (2, 5), 19, 7),
            ((2, 5), (3, 5), 20, 7),
            ((3, 5), (4, 7), 20, 6),
            ((4, 7), (4, 14), 21, 6),
            ((4, 14), (8, 28), 21, 5),
            ((8, 28), (9, 3), 21, 6),
            ((9, 3), (10, 13), 20, 6),
            ((10, 13), (11, 2), 19, 6),
            ((11, 2), (12, 31), 19, 7)
        ]

        # 브랜치와 패턴 그룹 매핑
        branch_group_map = {
            pattern_group_1: {'A', 'B', 'C', 'F'},
            pattern_group_2: {'D', 'E', 'G', 'H', 'I', 'J', 'K', 'S'},
            pattern_group_3: {'L', 'M', 'N'},
            pattern_group_O: {'O'},
            pattern_group_P: {'P'},
            pattern_group_Q: {'Q'},
            pattern_group_R: {'R'},
        }

        # 반복 처리
        for pattern_group, branches in branch_group_map.items():
            df = self._fill_si_by_pattern(df, branches, pattern_group)

        # 2. 'ta', 'wd', 'ws', 'rn_day', 'rn_hr1', 'hm' 변수
        # 1)장시간 결측
        vars_to_check = ['ta', 'wd', 'ws', 'rn_day', 'rn_hr1', 'hm','si']

        summaries = []
        for var in vars_to_check:
            summary = self._compute_null_streaks(df, var)
            summaries.append(summary)

        # 하나로 합치고, 중복 제거
        full_summary = pd.concat(summaries, ignore_index=True)
        unique_summary = full_summary.drop_duplicates(subset=['branch_id', 'start_time', 'end_time'])

        for var in vars_to_check:
            self._fill_long_term_missing_with_cluster_avg(self, df, summaries, var)

        # 2) 단시간 결측

        # 변수 리스트 (결측값 있을만한 변수들)
        vars_to_impute = ['ta', 'wd', 'ws', 'rn_day', 'rn_hr1', 'hm', 'ta_chi','si']

        # 각각 "클러스터별로" MICE 적용
        df = self._apply_mice(df, vars_to_impute)

        return df
    

    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        # 시간 관련 파생 변수 (train_heattm 컬럼이 있다고 가정)

        df['year'] = df['tm'].dt.year
        df['month'] = df['tm'].dt.month
        df['day'] = df['tm'].dt.day
        df['hour'] = df['tm'].dt.hour

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
        
        #풍향그룹
        df['wd_group'] = df['wd'].apply(self._categorize_wind_direction)

        #풍속그룹
        df['ws_group'] = df['ws'].apply(self._categorize_ws)

        #이슬점
        df['dew_point'] = df.apply(lambda row: self._dew_point(row['ta'], row['hm']), axis=1)
        
        #lag 변수 추가
        df = self._add_lag_features(df, lag=1)

        #이동평균
        target_cols = ['wd', 'ws', 'rn_hr1', 'rn_day']
        df = self._add_moving_averages(df, target_cols, window_size=3)

        #trend ,계절성변수
        daily_avg = (
        df.copy()
          .assign(date=lambda x: x['tm'].dt.date)
          .groupby('date')['heat_demand']
          .mean()
        )

        # STL 분해
        result = sm.tsa.seasonal_decompose(daily_avg, model='additive', period=365)
        df['trend'] = result.trend
        df['seasonal'] =result.seasonal
        df['residual'] = result.resid
        
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
        """수치형 변수 스케일링 (타겟 및 불필요한 컬럼 제외)"""
        # 전체 수치형 변수 추출
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        # 스케일링 대상에서 제외할 변수들
        exclude_cols = ['id', 'cluster_id', 'heat_demand', 'log_heat_demand']
        scale_cols = [col for col in numerical_cols if col not in exclude_cols]

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
        '''
        퓨리에 분석은 시계열 신호에서 어떤 주파수(주기)가 강한지 확인하여 데이터 특성을 파악하는 도구이다. 

        일반적인 흐름은 다음과 같다.
        1) 퓨리에 변환(FFT) → 주파수별 성분 분석 → 시각화
        2) 필요하면 원하는 주파수(진폭(스펙트럼) 그래프에서 높게 튀어오른 주파수 = 의미 있는 주파수) 성분만 살리고, 
            노이즈(원하지 않는 주파수) 제거 → 주파수 필터링
        3) 필터링된 신호를 역변환(iFFT) → 시간 영역의 깨끗한 신호 얻기
        4) 이렇게 얻은 신호를 원본 데이터프레임에 새 컬럼으로 넣어 활용
        '''
        #  1. 원본 시간 단위 데이터 처리
        # =====================
        df_copy = df.copy()
        df_copy['tm'] = pd.to_datetime(df_copy['tm'])  # datetime 변환

        # 원본 tm 따로 복사 (필요 시)
        tm_col = df_copy['tm'].copy()

        df_copy = df_copy.set_index('tm')              # tm을 인덱스로 설정

        # 일 평균 시계열 (STL용)
        daily_avg = df_copy['heat_demand'].resample('D').mean()

        # 시각 단위 원본도 저장 (tm 칼럼 복구)
        df_copy = df_copy.reset_index()  # tm을 다시 칼럼으로

        # =====================
        # 2. STL 분해
        # =====================
        period = 30  # 월간 계절성 가정

        result = sm.tsa.seasonal_decompose(daily_avg, model='additive', period=period)

        # STL 분해 후: NaN 포함된 resid
        resid = result.resid

        # 선형 보간 + 앞뒤 결측 보완
        resid_filled = resid.interpolate(method='time').ffill().bfill()  # 시간기반 보간 + 앞뒤 결측 처리

        df = self._process_fft_filtering(resid_filled, df)

        """시계열 특성 추가"""
        # 시간 기반 순환 특성 (sin, cos 변환)
        df['hour'] = df['tm'].dt.hour
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

    '''퓨리에에서 사용) fft 관련함수'''
    def _apply_fft(self, series):
        series_clean = series.dropna()
        centered = series_clean - series_clean.mean()
        fft_result = np.fft.fft(centered)
        freqs = np.fft.fftfreq(len(centered))
        return freqs, fft_result, series_clean.index  # 시계열 인덱스 반환

    def _filter_fft(self,freqs, fft_vals, low_cutoff=0.0, high_cutoff=0.15):
        filtered_fft = np.zeros_like(fft_vals)
        for i, freq in enumerate(freqs):
            if low_cutoff <= abs(freq) <= high_cutoff:
                filtered_fft[i] = fft_vals[i]
        return filtered_fft

    def _apply_ifft(self,filtered_fft):
        return np.fft.ifft(filtered_fft).real

    '''퓨리에에서 사용) Resid를 시각 단위로 확장하는 함수'''
    def _expand_residual_to_hourly(self, resid_series, full_df):
        """
        일 단위 residual을 해당 일의 모든 시간 단위 행에 복제해서 align
        """
        resid_df = resid_series.to_frame(name='cleaned_residual').copy()
        resid_df.index = pd.to_datetime(resid_df.index)
        resid_df['date'] = resid_df.index.date

        full_df = full_df.copy()
        full_df['date'] = pd.to_datetime(full_df['tm']).dt.date  # tm 칼럼 기준

        merged = full_df.merge(resid_df, on='date', how='left')
        merged = merged.drop(columns='date')  # 중간 컬럼 제거

        return merged
    '''퓨리에에서 사용) 전체 프로세스 실행 함수'''
    def _process_fft_filtering(self, daily_resid, full_df):
        # FFT
        freqs, fft_vals, idx = self._apply_fft(daily_resid)
        filtered_fft = self._filter_fft(freqs, fft_vals)
        cleaned_signal = self._apply_ifft(filtered_fft)
        
        # Cleaned 시리즈 생성
        cleaned_signal_series = pd.Series(cleaned_signal, index=idx, name='cleaned_residual')
        
        # Resid를 시간 단위로 확장
        full_df_with_resid = self._expand_residual_to_hourly(cleaned_signal_series, full_df)

        return full_df_with_resid


    def _basic_process(self, df : pd.DataFrame, is_training) -> pd.DataFrame:
        '''기본적인 전처리들'''
      # 날짜 데이터를 데이트타임으로 변경
        df['tm'] = pd.to_datetime(df['tm'].astype(str), format="%Y%m%d%H")

        # 263069개의 결측치 -> na로 대체
        df = df.replace(-99, np.nan)

        # 풍향 -9.9 값을 NaN으로 변경
        df['wd'] = df['wd'].replace(-9.9, np.nan)

        
    def _compute_null_streaks(self, df : pd.DataFrame, column_name) -> pd.DataFrame:
        """
        특정 column에 대해 branch별 연속 결측 구간을 계산하는 함수
        """
        result = []

        for branch in df['branch_id'].unique():
            df_branch = df[df['branch_id'] == branch].copy()

            # 결측 여부 마스크
            null_mask = df_branch[column_name].isnull()

            # 연속 구간 그룹 번호 부여
            group = (null_mask != null_mask.shift()).cumsum()

            # 결측만 필터링
            df_null = df_branch[null_mask].copy()
            df_null['group'] = group[null_mask]

            # 그룹별 통계
            summary = df_null.groupby('group').agg({
                'tm': ['count', 'min', 'max']
            }).reset_index()
            summary.columns = ['group', 'length', 'start_time', 'end_time']
            summary['branch_id'] = branch
            summary['feature'] = column_name
            summary['duration_days'] = (summary['end_time'] - summary['start_time']).dt.days + 1
            summary = summary[summary['length'] > 24]  # 장기 결측만

            result.append(summary)

        return pd.concat(result, ignore_index=True)
    
    
    def _fill_long_term_missing_with_cluster_avg(self, df: pd.DataFrame, summary_df, variable)-> pd.DataFrame:
        '''장시간 결측 채우기'''
        for idx, row in summary_df.iterrows():
            branch = row['branch_id']
            start_time = row['start_time']
            end_time = row['end_time']

            print(f"[{idx+1}/{len(summary_df)}] 처리 중: branch = {branch}, 기간 = {start_time} ~ {end_time}")


            # 결측 구간에 해당하는 tm 범위
            mask_time = (df['tm'] >= start_time) & (df['tm'] <= end_time)
            
            # 결측인 행들 (해당 branch, 해당 변수에서 null)
            mask_missing = mask_time & (df['branch_id'] == branch) & (df[variable].isnull())
            
            # 같은 시간대, 같은 클러스터에서 결측이 아닌 값들의 평균 구하기
            # 동일 시간대, 같은 변수, null 아닌 값만 필터링, branch는 결측 branch 제외
            avg_values = []
            for t in df.loc[mask_time, 'tm'].unique():
                mask_same_time = (df['tm'] == t) & (df[variable].notnull()) & (df['branch_id'] != branch)
                mean_val = df.loc[mask_same_time, variable].mean()
                avg_values.append((t, mean_val))
            
            # 평균값으로 채우기
            for t, val in avg_values:
                if pd.notnull(val):
                    df.loc[mask_missing & (df['tm'] == t), variable] = val

            print("결측 구간 처리 완료.")
        
        return df
    
    
    def _apply_mice(self, df, vars_to_impute)-> pd.DataFrame:
        '''단시간결측, mice방법론'''
        imputer = IterativeImputer(random_state=42)
        
        # imputer는 수치형 데이터만 다루므로, 수치형 컬럼만 선택
        df_impute = df[vars_to_impute]
        
        # fit_transform 후 결과는 numpy array
        imputed_array = imputer.fit_transform(df_impute)
        
        # 다시 DataFrame으로 변환
        df_imputed = df.copy()
        df_imputed[vars_to_impute] = imputed_array
        
        return df_imputed
    
    def _categorize_wind_direction(self, degree):
        '''풍향 4분할 - 동(45~135), 남(135~225), 서(225~315), 북(그 외)'''
        if 45 <= degree < 135:
            return 'E'  # 동
        elif 135 <= degree < 225:
            return 'S'  # 남
        elif 225 <= degree < 315:
            return 'W'  # 서
        else:
            return 'N'  # 북
        
    def _categorize_ws(self, ws):
        '''풍속기반 그룹화'''
        if ws < 1:
            return '정지'
        elif ws < 4:
            return '약풍'
        elif ws < 9:
            return '중풍'
        else:
            return '강풍'
    
    def _dew_point(self, temp, humidity):
        '''상대습도 기반 이슬점 온도 계산'''
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100)
        dp = (b * alpha) / (a - alpha)
        return dp
    
    def _add_lag_features(self, df, lag=1)-> pd.DataFrame:
        '''수치형 변수에 lag 변수 추가'''    
        # 사용할 변수와 lag 시기 정의
        target_vars = ['ta', 'hm', 'si', 'ta_chi', 'heat_demand']
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

        
    def _fill_si_by_pattern(self, df: pd.DataFrame, branches: set, patterns: list):
        '''해당 패턴일 때 & si변수가 null값이면 0으로 만드는 함수'''
        df = df.copy()

        # 날짜+시간 추출
        df['month'] = df['tm'].dt.month
        df['day'] = df['tm'].dt.day
        df['hour'] = df['tm'].dt.hour

        mask = pd.Series(False, index=df.index)

        for (start_md, end_md, night_start, night_end) in patterns:
            start_m, start_d = start_md
            end_m, end_d = end_md

            if start_md <= end_md:
                date_mask = (
                    ((df['month'] > start_m) | ((df['month'] == start_m) & (df['day'] >= start_d))) &
                    ((df['month'] < end_m) | ((df['month'] == end_m) & (df['day'] <= end_d)))
                )
            else:
                # 연말-연초 넘는 경우
                date_mask = (
                    ((df['month'] > start_m) | ((df['month'] == start_m) & (df['day'] >= start_d))) |
                    ((df['month'] < end_m) | ((df['month'] == end_m) & (df['day'] <= end_d)))
                )

            # 시간 조건
            if night_start < night_end:
                time_mask = (df['hour'] >= night_start) & (df['hour'] < night_end)
            else:
                time_mask = (df['hour'] >= night_start) | (df['hour'] < night_end)

            mask |= date_mask & time_mask

        # 조건 만족시 si = 0
        df.loc[df['branch_id'].isin(branches) & df['si'].isna() & mask, 'si'] = 0

        # 정리
        return df.drop(['month', 'day', 'hour'], axis=1)
