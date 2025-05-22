from sklearn.model_selection import train_test_split

def data_split(split_type, train_df, target_column):
    if split_type == "random":
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        x_train, x_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"랜덤 분할 완료: 학습 {len(x_train)}개, 검증 {len(x_valid)}개")
    elif split_type == "time":
        holdout_start = 2023070101
        holdout_end = 2023123123
        is_holdout = ((train_df["tm"] >= holdout_start) & 
                      (train_df["tm"] <= holdout_end))
        x_train = train_df[~is_holdout].drop(columns=[target_column])
        y_train = train_df[~is_holdout][target_column]
        x_valid = train_df[is_holdout].drop(columns=[target_column])
        y_valid = train_df[is_holdout][target_column]
        print(f"시간 기반 분할 완료: 학습 {len(x_train)}개, 검증 {len(x_valid)}개")
    else:
        raise ValueError(f"지원하지 않는 분할 방식입니다: {split_type}. 'random' 또는 'time' 중에서 선택하세요.")
    return x_train, x_valid, y_train, y_valid
