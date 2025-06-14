import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL

class WeatherDataPreprocessor:
    """날씨 데이터 전처리를 위한 클래스"""
    
    def __init__(self):
        self.lag_cols = ['ta', 'ws', 'rn_hr1', 'hm', 'si', 'ta_chi']
        self.rolling_cols = ['ta', 'ws', 'rn_hr1', 'hm', 'si', 'ta_chi']
        self.lag_hours = [1, 3, 24]
        self.rolling_hours = [3, 24]
        self.breakpoints = {
            'A': (-4.60, 16.51), 'B': (-7.52, 16.92), 'C': (-4.19, 18.07),
            'D': (-4.13, 14.22), 'E': (10.86, 23.04), 'F': (-7.58, 13.99),
            'G': (-7.50, 17.02), 'H': (-5.72, 16.38), 'I': (-6.26, 15.77),
            'J': (13.76, 21.94), 'K': (-4.70, 17.85), 'L': (-0.70, 17.96),
            'M': (-0.30, 18.05), 'N': (-2.58, 17.37), 'O': (-5.00, 17.04),
            'P': (-7.30, 17.81), 'Q': (-9.10, 17.09), 'R': (-6.10, 18.80),
            'S': (-8.83, 17.29)
        }
        
        # 클러스터 매핑
        self.cluster_map = {
            'E':0, 'F':0, 'I':0, 'J':0, 'K':0, 'N':0, 'O':0, 'Q':0,
            'B':1, 'C':1, 'G':1,
            'A':2, 'D':2, 'H':2, 'P':2,
            'L':3, 'M':3, 'R':3, 'S':3
        }
        
        # 클러스터별 lag 설정
        self.cluster_lag_config = {
            0: {
                'ta': list(range(1, 4)),
                'hm': [1, 2],
                'rn_day': [10, 11],
            },
            1: {
                'ta': [1, 2, 3],
                'ws': [3, 4, 5],
            },
            3: {
                'ta': [1, 2],
                'hm': [1],
                'rn_hr1': [22, 23],
            },
            2: {
                'summer': {
                    'ta': list(range(1, 4)) + list(range(8, 12)),
                    'rn_day': list(range(10, 25)),
                    'rn_hr1': list(range(22, 25)),
                    'hm': list(range(1, 4)) + list(range(7, 25)),
                    'si': list(range(1, 5)) + list(range(18, 25)),
                    'ta_chi': [1, 2, 3, 23, 24],
                    'year': list(range(1, 6)),
                    'month': list(range(1, 6)),
                    'day': list(range(1, 8)) + list(range(20, 25)),
                    'day_of_week': list(range(1, 17)) + list(range(20, 25)),
                },
                'non_summer': {
                    'ta': list(range(1, 20)) + [22, 23, 24],
                    'wd': [1] + list(range(5, 20)),
                    'ws': list(range(3, 25)),
                    'rn_day': list(range(9, 20)),
                    'rn_hr1': [1, 3, 4, 5, 20, 21, 22, 23, 24],
                    'hm': list(range(5, 21)) + [24],
                    'si': list(range(1, 8)) + list(range(9, 22)),
                    'ta_chi': list(range(1, 25)),
                    'month': list(range(13, 25)),
                    'day': list(range(8, 21)),
                    'hour': [17],
                    'day_of_week': list(range(1, 10)) + list(range(18, 25)),
                    'wd_rad': [1] + list(range(5, 20)),
                    'wd_sin': [1, 2] + list(range(6, 15)),
                    'wd_cos': list(range(1, 18)),
                }
            }
        }
        
        # 야간 시간 규칙
        self._init_night_rules()

    def _init_night_rules(self):
        """야간 시간 규칙 초기화"""
        # 공통 규칙 정의
        common_night_rules_1 = [
            ((1, 1), (2, 6), 19, 7),
            ((2, 6), (3, 4), 20, 7),
            ((3, 4), (4, 7), 20, 6),
            ((4, 7), (4, 13), 21, 6),
            ((4, 13), (8, 28), 21, 5),
            ((8, 28), (9, 3), 21, 6),
            ((9, 3), (10, 12), 20, 6),
            ((10, 12), (11, 3), 19, 6),
            ((11, 3), (12, 31), 19, 7)
        ]

        common_night_rules_2 = [
            ((1, 1), (2, 5), 19, 7),
            ((2, 5), (3, 4), 20, 7),
            ((3, 4), (4, 7), 20, 6),
            ((4, 7), (4, 14), 21, 6),
            ((4, 14), (8, 28), 21, 5),
            ((8, 28), (9, 3), 21, 6),
            ((9, 3), (10, 13), 20, 6),
            ((10, 13), (11, 4), 19, 6),
            ((11, 4), (12, 31), 19, 7)
        ]

        common_night_rules_3 = [
            ((1, 1), (2, 10), 19, 7),
            ((2, 10), (2, 25), 20, 7),
            ((2, 25), (4, 9), 20, 6),
            ((4, 9), (4, 19), 20, 5),
            ((4, 19), (8, 26), 21, 5),
            ((8, 26), (9, 4), 20, 5),
            ((9, 4), (10, 7), 20, 6),
            ((10, 7), (11, 13), 19, 6),
            ((11, 13), (12, 31), 19, 7)
        ]

        common_night_rules_4 = [
            ((1, 1), (2, 6), 19, 7),
            ((2, 6), (3, 2), 20, 7),
            ((3, 2), (4, 11), 20, 6),
            ((4, 11), (4, 13), 21, 6),
            ((4, 13), (8,30), 21, 5),
            ((8, 30), (9, 1), 21, 6),
            ((9, 1), (10, 11), 20, 6),
            ((10, 11), (11, 7), 19, 6),
            ((11, 7), (12, 31), 19, 7)
        ]

        common_night_rules_5 = [
            ((1, 1), (2, 6), 19, 7),
            ((2, 6), (3, 3), 20, 7),
            ((3, 3), (4, 10), 20, 6),
            ((4, 10), (4, 13), 21, 6),
            ((4, 13), (8,30), 21, 5),
            ((8, 30), (9, 2), 21, 6),
            ((9, 2), (10, 12), 20, 6),
            ((10, 12), (11, 6), 19, 6),
            ((11, 6), (12, 31), 19, 7)
        ]

        common_night_rules_6 = [
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

        common_night_rules_7 = [
            ((1, 1), (2, 1), 19, 7),
            ((2, 1), (3, 2), 20, 7),
            ((3, 2), (4, 9), 20, 6),
            ((4, 9), (4, 15), 21, 6),
            ((4, 15), (8, 26), 21, 5),
            ((8, 26), (9, 2), 21, 6),
            ((9, 2), (10, 14), 20, 6),
            ((10, 14), (11, 8), 19, 6),
            ((11, 8), (12, 31), 19, 7)
        ]

        # 브랜치별 야간 시간 규칙 딕셔너리
        self.branch_night_rules = {
            'A': common_night_rules_1,
            'B': common_night_rules_1,
            'C': common_night_rules_1,
            'D': common_night_rules_2,
            'E': common_night_rules_2,
            'F': common_night_rules_1,
            'G': common_night_rules_2,
            'H': common_night_rules_2,
            'I': common_night_rules_2,
            'J': common_night_rules_2,
            'K': common_night_rules_2,
            'L': common_night_rules_3,
            'M': common_night_rules_3,
            'N': common_night_rules_3,
            'O': common_night_rules_4,
            'P': common_night_rules_5,
            'Q': common_night_rules_6,
            'R': common_night_rules_7,
            'S': common_night_rules_2
        }

    def clean_column_names(self, df, prefix='train_heat.'):
        """컬럼명 정리"""
        df = df.copy()
        df.columns = [col.replace(prefix, "") for col in df.columns]
        return df

    def handle_datetime_conversion(self, df, time_col='tm'):
        """datetime 변환 처리"""
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col].astype(str), format="%Y-%m-%d %H:%M:%S")
        return df

    def handle_missing_values(self, df, is_test=False):
        """결측값 처리"""
        df = df.copy()
        
        # -99.0을 np.nan으로 변경
        cols_with_neg99 = [col for col in df.columns if (df[col] == -99.0).any()]
        if cols_with_neg99:
            print("'-99.0' 포함 열:", cols_with_neg99)
            df[cols_with_neg99] = df[cols_with_neg99].replace(-99.0, np.nan)
        
        # -9.9를 np.nan으로 변경 (특히 wd 컬럼)
        if 'wd' in df.columns:
            df['wd'] = df['wd'].replace(-9.9, np.nan)
        
        # 타겟 변수 heat_demand에 결측이 있는 행 제거
        if is_test is False and 'heat_demand' in df.columns:
            df = df.dropna(subset=['heat_demand']).reset_index(drop=True)
        
        return df

    def create_datetime_features(self, df, time_col='tm'):
        """시간 관련 특성 생성"""
        df = df.copy()
        
        # 기본 시간 특성
        df['year'] = df[time_col].dt.year
        df['quarter'] = df[time_col].dt.quarter
        df['month'] = df[time_col].dt.month
        df['day'] = df[time_col].dt.day
        df['hour'] = df[time_col].dt.hour
        df['date'] = df[time_col].dt.date
        df['day_of_week'] = df[time_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        df['weekofyear'] = df[time_col].dt.isocalendar().week
        
        # 3시간 단위 시간대 컬럼 생성
        df['hour_group'] = (df[time_col].dt.hour // 3).astype(int)
        
        return df

    def apply_night_rules(self, df):
        """야간 시간 규칙 적용 (일사량 조정)"""
        df = df.copy()
        
        # 모든 브랜치에 대해 야간 시간 적용
        for branch, rules in self.branch_night_rules.items():
            is_branch = df['branch_id'] == branch

            for (start_m, start_d), (end_m, end_d), night_start, night_end in rules:
                in_period = (
                    ((df['month'] > start_m) | ((df['month'] == start_m) & (df['day'] >= start_d))) &
                    ((df['month'] < end_m) | ((df['month'] == end_m) & (df['day'] <= end_d)))
                )

                if night_start > night_end:  # 자정을 넘기는 경우
                    in_night = (df['hour'] >= night_start) | (df['hour'] <= night_end)
                else:
                    in_night = (df['hour'] >= night_start) & (df['hour'] < night_end)

                # 조건에 맞는 행에 대해 si 값 0으로 설정
                mask = is_branch & in_period & in_night
                df.loc[mask, 'si'] = 0
        
        return df

    def add_cluster_lags(self, df):
        """클러스터별 lag 변수 추가"""
        cluster_dfs = {}
        
        for branch_id, group in df.groupby('branch_id'):
            cluster = self.cluster_map.get(branch_id, -1)
            if cluster not in self.cluster_lag_config:
                continue

            # 클러스터 2 여부 확인 (기존 코드 유지)
            if cluster == 2:
                summer_mask = group['month'].between(6, 9)
                summer_group = group[summer_mask].copy()
                non_summer_group = group[~summer_mask].copy()

                if 'summer' in self.cluster_lag_config[cluster] and not summer_group.empty:
                    summer_processed = self._process_group(summer_group, self.cluster_lag_config[cluster]['summer'])
                    cluster_dfs.setdefault('cluster2_summer', []).append(summer_processed)

                if 'non_summer' in self.cluster_lag_config[cluster] and not non_summer_group.empty:
                    non_summer_processed = self._process_group(non_summer_group, self.cluster_lag_config[cluster]['non_summer'])
                    cluster_dfs.setdefault('cluster2_non_summer', []).append(non_summer_processed)
            else:
                processed = self._process_group(group, self.cluster_lag_config[cluster])
                cluster_dfs.setdefault(f'cluster{cluster}', []).append(processed)

        # 최종 데이터프레임 생성 - 수정된 부분
        result = {}
        for cluster_name, parts in cluster_dfs.items():
            if parts:
                # 1. 먼저 concat
                combined = pd.concat(parts, ignore_index=True)
                
                # 2. 정렬
                combined = combined.sort_values(['branch_id', 'tm']).reset_index(drop=True)
                
                # 3. 중복 제거 (tm + branch_id 기준)
                combined = combined.drop_duplicates(
                    subset=['tm', 'branch_id'],
                    keep='first'
                ).reset_index(drop=True)
                
                # 4. NaN이 있는 행 제거
                result[cluster_name] = combined.dropna().reset_index(drop=True)

        return result

    def _process_group(self, df, config):
        """그룹별 lag 처리"""
        lagged = [df]
        for base_col, lags in config.items():
            if base_col in df.columns:  # 컬럼이 존재하는 경우만 처리
                for lag in lags:
                    new_col = f"{base_col}_lag{lag}"
                    lagged_df = df[[base_col]].shift(lag).rename(columns={base_col: new_col})
                    lagged.append(lagged_df)
        return pd.concat(lagged, axis=1)

    def _handle_nan(self, df, parts):
        """NaN 처리"""
        if len(parts) > 0 and len(parts[0]) > 24:
            # 각 부분의 첫 24시간 데이터 캐싱
            cached_parts = [part.head(24) for part in parts]
            cached_data = pd.concat(cached_parts)
            return pd.concat([cached_data, df]).sort_index().iloc[:-24]
        return df

    def data_clipping(self, df):
        """데이터 범위 제한"""
        df = df.copy()
        # 풍향 범위는 0~360도
        if 'wd' in df.columns:
            df['wd'] = df['wd'] % 360
        # 일사량, 풍속, 강수량 등은 0 이상만 허용
        for col in ['si', 'ws', 'rn_day', 'rn_hr1']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        return df

    def create_lag_rolling_features(self, df):
        """lag 및 rolling 변수 생성"""
        dfs = []
        
        for branch, group in df.groupby('branch_id'):
            df_branch = group.sort_values('tm').copy()
            
            # lag 변수 생성
            for col in self.lag_cols:
                if col in df_branch.columns:
                    for lag in self.lag_hours:
                        new_col = f'{col}_lag{lag}'
                        df_branch[new_col] = df_branch[col].shift(lag)
            
            # 이동평균 변수 생성 (과거 값만 반영) - 3일과 7일 추가
            for col in self.rolling_cols:
                if col in df_branch.columns:
                    for window in self.rolling_hours:
                        new_col = f'{col}_roll{window}'
                        df_branch[new_col] = df_branch[col].shift(1).rolling(window=window).mean()
                    
                    # 3일 이동평균 추가
                    df_branch[f'{col}_roll72'] = df_branch[col].shift(1).rolling(window=72).mean()  # 3일 = 72시간
                    # 7일 이동평균 추가
                    df_branch[f'{col}_roll168'] = df_branch[col].shift(1).rolling(window=168).mean()  # 7일 = 168시간
            
            dfs.append(df_branch)
        
        return pd.concat(dfs, axis=0).sort_values(['branch_id', 'tm']).reset_index(drop=True)

    def create_rain_intensity(self, df):
        """강우 강도 생성"""
        df = df.copy()
        if 'rn_hr1' in df.columns:
            conditions = [
                df['rn_hr1'] < 3,
                (df['rn_hr1'] >= 3) & (df['rn_hr1'] < 15),
                (df['rn_hr1'] >= 15) & (df['rn_hr1'] < 30),
                df['rn_hr1'] >= 30
            ]
            choices = ['weak', 'normal', 'strong', 'very strong']
            df['rain_intensity'] = np.select(conditions, choices, default='unknown')
        return df

    def create_temperature_flags(self, df):
        """폭염 및 한파 플래그 생성"""
        df = df.copy()
        if 'ta' in df.columns and 'date' in df.columns:
            daily_temp = df.groupby('date').agg(
                ta_max=('ta', 'max'), 
                ta_min=('ta', 'min')
            ).reset_index()
            
            daily_temp['heatwave'] = (daily_temp['ta_max'] >= 33).astype(int)
            daily_temp['coldwave'] = (daily_temp['ta_min'] <= -12).astype(int)
            
            df = df.merge(daily_temp[['date', 'heatwave', 'coldwave']], on='date', how='left')
        return df

    def create_time_features(self, df):
        """시간 기반 순환 특성 생성 (확장된 버전)"""
        df = df.copy()
        
        # 기본 시간 특성
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 추가 시간 특성들
        if 'tm' in df.columns:
            # 주 내 시간 (0~167)
            df['hour_of_week'] = df['tm'].dt.dayofweek * 24 + df['hour']
            df['day_of_month'] = df['tm'].dt.day
            
            # 일주일 단위 순환
            df['weekhour_sin'] = np.sin(2 * np.pi * df['hour_of_week'] / 168)
            df['weekhour_cos'] = np.cos(2 * np.pi * df['hour_of_week'] / 168)
            
            # 월 내 일자 순환 (30일 기준)
            df['monthday_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 30)
            df['monthday_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 30)
        
        return df

    def create_temperature_features(self, df):
        """온도 관련 특성 생성"""
        df = df.copy()
        if 'ta_chi' in df.columns and 'ta' in df.columns:
            # 체감온도와 실제 온도의 차이
            df['ta_diff_chi'] = df['ta_chi'] - df['ta']
        if 'ta' in df.columns:
            # 냉난방 필요 정도
            df['heating_degree'] = (18 - df['ta']).clip(lower=0)
            df['cooling_degree'] = (df['ta'] - 24).clip(lower=0)
        return df

    def create_wind_features(self, df):
        """풍향 특성 생성 (확장된 버전)"""
        df = df.copy()
        if 'wd' in df.columns:
            df['wd_rad'] = np.deg2rad(df['wd'])
            df['wd_sin'] = np.sin(df['wd_rad'])
            df['wd_cos'] = np.cos(df['wd_rad'])
            
            # 풍향 그룹 추가
            df['wd_group'] = df['wd'].apply(self.categorize_wind_direction)
        
        if 'ws' in df.columns:
            # 풍속 그룹 추가
            df['ws_group'] = df['ws'].apply(self.categorize_ws)
        
        return df

    def categorize_wind_direction(self, degree):
        """풍향을 카테고리로 변환"""
        if pd.isna(degree):
            return 'Unknown'
        if 45 <= degree < 135:
            return 'E'
        elif 135 <= degree < 225:
            return 'S'
        elif 225 <= degree < 315:
            return 'W'
        else:
            return 'N'

    def categorize_ws(self, ws):
        """풍속을 카테고리로 변환"""
        if pd.isna(ws):
            return 'Unknown'
        if ws < 1:
            return '정지'
        elif ws < 4:
            return '약풍'
        elif ws < 9:
            return '중풍'
        else:
            return '강풍'

    def create_day_flag(self, df):
        """주간 플래그 생성"""
        df = df.copy()
        if 'hour' in df.columns:
            df['day_flag'] = df['hour'].apply(lambda x: 1 if 12 <= x <= 18 else 0)
        return df

    def classify_season_v2(self, month):
        """계절 분류"""
        if month in [6, 7, 8, 9]:
            return 'summer'
        elif month in [3, 11]:
            return 'early_winter'
        elif month in [4, 5, 10]:
            return 'mid_season'
        elif month in [1, 2, 12]:
            return 'winter'

    def create_season_group(self, df):
        """계절 그룹 생성"""
        df = df.copy()
        if 'month' in df.columns:
            df['season_group'] = df['month'].apply(self.classify_season_v2)
        return df

    def assign_ta_zone(self, row):
        """온도 구간 할당"""
        branch = row['branch_id']
        ta = row['ta']
        
        if branch in self.breakpoints:
            x0, x1 = self.breakpoints[branch]
            if ta < x0:
                return 'low'
            elif ta < x1:
                return 'mid'
            else:
                return 'high'
        else:
            return 'unknown'

    def create_ta_zone(self, df):
        """온도 구간 생성"""
        df = df.copy()
        if 'branch_id' in df.columns and 'ta' in df.columns:
            df['ta_zone'] = df.apply(self.assign_ta_zone, axis=1)
        return df

    def create_cumulative_si(self, df):
        """누적 일사량 생성"""
        df = df.copy()
        if 'si' in df.columns and 'branch_id' in df.columns and 'date' in df.columns:
            df['cumulative_si'] = df.groupby(['branch_id', 'date'])['si'].cumsum()
        return df

    def dew_point(self, temp, humidity):
        """이슬점 계산"""
        if pd.isna(temp) or pd.isna(humidity) or humidity <= 0:
            return np.nan
        a, b = 17.27, 237.7
        alpha = ((a * temp) / (b + temp)) + np.log(humidity / 100)
        return (b * alpha) / (a - alpha)

    def create_dew_point(self, df):
        """이슬점 특성 생성"""
        df = df.copy()
        if 'ta' in df.columns and 'hm' in df.columns:
            df['dew_point'] = df.apply(lambda row: self.dew_point(row['ta'], row['hm']), axis=1)
        return df

    def create_stl_decomposition(self, df):
        """STL 분해를 통한 계절성 변수 생성"""
        df = df.copy()
        result_list = []
        
        for branch_id, group in df.groupby('branch_id'):
            if 'ta' in group.columns and len(group) >= 48:
                try:
                    series = group.set_index('tm')['ta']  # 시간 인덱스 설정
                    stl = STL(series, period=24).fit()
                    
                    group = group.copy()
                    group['trend'] = stl.trend.values
                    group['seasonal'] = stl.seasonal.values
                    group['residual'] = stl.resid.values
                    
                except Exception as e:
                    print(f"STL decomposition failed for branch {branch_id}: {e}")
                    # STL 실패시 기본값으로 채움
                    group['trend'] = group['ta']
                    group['seasonal'] = 0
                    group['residual'] = 0
            else:
                # 데이터 부족시 기본값
                group['trend'] = group['ta'] if 'ta' in group.columns else 0
                group['seasonal'] = 0
                group['residual'] = 0
                
            result_list.append(group)
        
        return pd.concat(result_list).sort_values(['branch_id', 'tm']).reset_index(drop=True)
                
    def create_di_features(self, df):
        """불쾌지수"""
        # 여름 여부 (boolean으로 만들기)
        df['DI'] = (
            0.81 * df['ta'] +
            0.01 * df['hm'] * (0.99 * df['ta'] - 14.3) +
            46.3
        )

        df['is_summer'] = (df['season_group'] == 'Summer').astype(int)

        # 주요 지점 여부
        df['is_key_branch'] = df['branch_id'].apply(lambda x: 1 if x in ['K','E','J','C','D','P','R','S'] else 0)

        # 삼중 상호작용항: 여름 & K/E/J/C/D/P/R/S & 불쾌지수
        df['DI_interaction'] = df['DI'] * df['is_key_branch'] * df['is_summer']
        return df
                
                
    def preprocess_data(self, df, is_test=False, column_prefix='train_heat.'):
        """전체 전처리 파이프라인"""
        print("데이터 전처리 시작...")
        
        # 1. 컬럼명 정리
        df = self.clean_column_names(df, prefix=column_prefix)
        
        # 2. datetime 변환
        if 'tm' in df.columns:
            df = self.handle_datetime_conversion(df)
        
        # 3. 결측값 처리
        df = self.handle_missing_values(df, is_test)
        
        # 4. datetime 기반 특성 생성
        if 'tm' in df.columns:
            df = self.create_datetime_features(df)
        
        # 5. 기본 클리핑
        df = self.data_clipping(df)
        
        # 6. 야간 시간 규칙 적용
        if 'si' in df.columns and 'branch_id' in df.columns:
            df = self.apply_night_rules(df)
        
        # 7. 테스트 데이터의 경우 결측치 처리
        if is_test:
            na_columns = df.columns[df.isna().any()].tolist()
            if na_columns:
                df[na_columns] = df[na_columns].fillna(method='ffill')
                df[na_columns] = df[na_columns].fillna(0)
        
        # 8. lag/rolling 변수 생성         
        # 클러스터별 lag 생성 - 딕셔너리 반환
        cluster_results = self.add_cluster_lags(df)
        
        # 각 클러스터 데이터에 대해 나머지 전처리 진행
        processed_clusters = {}
        for cluster_name, cluster_df in cluster_results.items():
            print(f"{cluster_name} 전처리 중...")
            
            # 9-13. 특성 생성
            cluster_df = self.create_rain_intensity(cluster_df)
            cluster_df = self.create_temperature_flags(cluster_df)
            cluster_df = self.create_time_features(cluster_df)
            cluster_df = self.create_temperature_features(cluster_df)
            cluster_df = self.create_wind_features(cluster_df)
            cluster_df = self.create_day_flag(cluster_df)
            cluster_df = self.create_season_group(cluster_df)
            cluster_df = self.create_ta_zone(cluster_df)
            cluster_df = self.create_cumulative_si(cluster_df)
            
            # 10. 이슬점 특성 생성
            cluster_df = self.create_dew_point(cluster_df)
            
            # 11. STL 분해
            if 'ta' in cluster_df.columns:
                self.create_stl_decomposition(cluster_df)
            
            # 12. 불쾌지수 특성 생성
            cluster_df = self.create_di_features(cluster_df)
            
            # 13. 로그 변환
            cluster_df['heat_demand_log'] = np.log1p(cluster_df['heat_demand']) if 'heat_demand' in cluster_df.columns else None
            
            processed_clusters[cluster_name] = cluster_df
            
            # 14. 불필요한 컬럼 제거
            drop_cols = ['season', 'wd', 'quarter', 'day_of_week', 'date','DI','is_summer','is_key_branch']
            existing_drop_cols = [col for col in drop_cols if col in cluster_df.columns]
            if existing_drop_cols:
                cluster_df = cluster_df.drop(columns=existing_drop_cols)
        
        print("클러스터별 데이터 전처리 완료!")
        return processed_clusters
            