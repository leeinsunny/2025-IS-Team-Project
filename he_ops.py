from heaan_stat import Context, HESeries

#1. 동형암호 컨텍스트 생성 함수
def make_context(key_dir="./keys"):
    return Context(key_dir_path=key_dir, generate_keys=False)

#2. 암호문 상태에서 내적(dot product) 계산 함수
def encrypted_dot(context, x, w):
    # 1) 입력 벡터 암호화
    x_he = HESeries(context, x, encrypt=True)

    # 2) 가중치 벡터 암호화
    w_he = HESeries(context, w.tolist())

    # 3) 슬롯별 곱셈 (암호문 상태)
    prod = x_he * w_he
    
    # 4) 암호문 상태에서 합산 (예: 모든 슬롯 합산)
    dot_product_encrypted = prod.sum()  # HEaaN에서 지원하는 암호문 합산 함수 사용

    # 5) 평문 벡터 합산하여 dot product 반환
    return dot_product_encrypted

  