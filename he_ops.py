from heaan_stat import Context, HESeries, Block

#1. 동형암호 컨텍스트 생성 함수
def make_context(key_dir="./keys"):
    return Context(key_dir_path=key_dir, generate_keys=False)

#2. 암호문 상태에서 내적(dot product) 계산 함수
def encrypted_dot(context, x, w, b):
    # 1) 입력 벡터 암호화
    x_he = Block(context, data = x, encrypted=True)
    print('입력벡터 암호화',x_he)

    # 2) 가중치 벡터 암호화
    w_he = Block(context, data = w.tolist(), encrypted=True)
    print('가중치 벡터 암호화',w_he)

    # bias float 암호화
    b_he = Block(context, data =[b,0], encrypted=True)
    print('bias 암호화',b_he)

    #내적을 구현하기 위해서 곱셈 후 덧셈을 해야함 
    prod_tmp =x_he * w_he
    print('mult = ', prod_tmp)
    print()
    print()
    print()
    print()
    # 3) 슬롯별 덧셈 (암호문 상태)
    prod = prod_tmp.sum(axis=1, direction=0, unit_shape=(16, 2048))
    print('sum = ',prod)
    print()
    print()
    print()
    print()
    # 4) 암호문 상태에서 합산 (예: 모든 슬롯 합산)
    z_enc = prod + b_he
    print(f"{b = }")
    print()
    print()
    print()
    print()

    print(f"{z_enc = }")
    print()
    print()
    print()
    print()
 

    return z_enc

  