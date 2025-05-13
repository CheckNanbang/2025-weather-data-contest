import argparse

def parse_args():
    """
    명령줄 인자를 파싱하는 함수.
    
    Returns:
        argparse.Namespace: 파싱된 명령줄 인자
    """
    parser = argparse.ArgumentParser(description="회귀 모델 학습 및 예측 프로그램")

    parser.add_argument("--dataset", type=str, default="heat", 
                        help="사용할 데이터셋 이름 (기본값: heat)")
    parser.add_argument("--model", type=str, default="XGB", choices=["LGBM", "XGB"],
                        help="사용할 모델 이름 (기본값: XGB)")
    parser.add_argument("--split_type", type=str, default="time", choices=["random", "time"],
                        help="데이터 분할 방식 (기본값: time)")
    parser.add_argument("--submit", action="store_true", 
                        help="제출용 예측 파일 생성 여부")
    parser.add_argument("--tune", action="store_true", 
                        help="Optuna로 하이퍼파라미터 튜닝 수행 여부")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Optuna 튜닝 시 시도할 횟수 (기본값: 20)")
    parser.add_argument("--plot", action="store_true",
                        help="학습 결과 시각화 여부")
    parser.add_argument("--params", nargs="*", 
                        help="key=value 형태로 직접 하이퍼파라미터 입력")
    
    return parser.parse_args()

def parse_params(param_list):
    """
    명령줄에서 전달된 파라미터를 파싱하는 함수.
    
    Args:
        param_list (list): 'key=value' 형태의 문자열 리스트
        
    Returns:
        dict: 파싱된 파라미터 딕셔너리
    """
    if not param_list:
        return {}
    
    params = {}
    for param in param_list:
        key, value = param.split('=')
        try:
            # 숫자인 경우 적절한 타입으로 변환
            params[key] = eval(value)
        except:
            # 문자열 등 다른 타입은 그대로 유지
            params[key] = value
    
    return params