# utils/data_split.py
from sklearn.model_selection import train_test_split

def data_split(split_type, train_df, target_column):
    """
    데이터셋을 지정된 방식에 따라 학습 및 검증 세트로 분할.

    Args:
        split_type (str): 데이터 분할 방식, "random" 또는 "time"
        train_df (pd.DataFrame): 분할할 데이터 프레임
        target_column (str): 예측 대상이 되는 타겟 컬럼 이름

    Returns:
        tuple: 학습 및 검증 세트로 분할된 입력과 타겟 (x_train, x_valid, y_train, y_valid)
    
    Raises:
        ValueError: 지원하지 않는 분할 방식이 제공될 경우
    """
    if split_type == "random":
        # 무작위 분할: 데이터를 랜덤하게 학습(80%)과 검증(20%) 세트로 분할
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, 
            y, 
            test_size=0.2,
            random_state=42
        )
        print(f"랜덤 분할 완료: 학습 {len(x_train)}개, 검증 {len(x_valid)}개")
        
    elif split_type == "time":
        # 시간 기반 분할: 특정 시간 범위를 검증 세트로 사용
        holdout_start = 2023070101  # 2023년 7월 1일 01시
        holdout_end = 2023123123    # 2023년 12월 31일 23시

        # 시간 기준 분할 조건
        is_holdout = ((train_df["train_heattm"] >= holdout_start) & 
                      (train_df["train_heattm"] <= holdout_end))
        
        # 학습 데이터: 홀드아웃 기간이 아닌 데이터
        x_train = train_df[~is_holdout].drop(columns=[target_column])
        y_train = train_df[~is_holdout][target_column]

        # 검증 데이터: 홀드아웃 기간의 데이터
        x_valid = train_df[is_holdout].drop(columns=[target_column])
        y_valid = train_df[is_holdout][target_column]
        
        print(f"시간 기반 분할 완료: 학습 {len(x_train)}개 ({train_df[~is_holdout]['train_heattm'].min()}-{train_df[~is_holdout]['train_heattm'].max()}), " 
              f"검증 {len(x_valid)}개 ({holdout_start}-{holdout_end})")
    else:
        raise ValueError(f"지원하지 않는 분할 방식입니다: {split_type}. 'random' 또는 'time' 중에서 선택하세요.")
        
    return x_train, x_valid, y_train, y_valid

