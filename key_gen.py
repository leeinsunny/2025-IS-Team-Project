import os
os.environ["HEAAN_TYPE"] = "pi" # use pi for using pi-heaan,you can use this for other ipynb files to using pi-heaan
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions

import numpy as np
import pandas as pd

import heaan_stat

from model_utils import load_scaler, load_model_parameters
from he_ops      import make_context, encrypted_dot


context = heaan_stat.Context(
    key_dir_path='./keys_stat',
    generate_keys=False,  # To use existing keys, set it to False or omit this
)

x_raw = [23.1, 120.0, 75.0, 88.5, 5.0, 1.0, 0.0, 1.0, 0.0, 0.0]
print(f"✅ 사용자 입력값: {x_raw}")

# -- 2. 스케일러 로드 및 정규화
scaler   = load_scaler("data/scaler.pkl")
x_scaled = scaler.transform([x_raw])[0]
print(f"✅ 정규화 결과: {x_scaled.tolist()}")

# -- 3. 학습된 모델 파라미터 불러오기 (가중치, 편향)
weights, bias = load_model_parameters(
    w_path="data/lr_weights.npy",
    b_path="data/lr_bias.npy"
)
dot_cipher = encrypted_dot(context, x_scaled.tolist(), weights)
print(f"✅  Dot product (암호화상태): {dot_cipher}")


z    = dot_cipher + bias
def encrypted_sigmoid(enc_val):
    x = enc_val
    x3 = x * x * x
    return 0.5 + (0.197 * x) - (0.004 * x3)
prob = encrypted_sigmoid(z)

print(prob)
prob_plain = prob.decrypt(False).to_series()
prob_val = prob_plain.iloc[0]

print(prob_plain)
pred = int(prob_val >= 0.5)
print(f"✅ 예측 확률: {prob_val:.4f}")
print(f"✅  예측 결과 (0=No, 1=Yes): {pred}")



#TODO
#14 - 33 ? 36? 까지는 프론트에서 처리
#37 - 41은 서버에서 연산
# 이후 서버연산값을 프론트에 넘겨서 여기서 복호화 
# 그리고 지금 생각해볼게 bias는 평문연산인데 dot_cipher가 암호문이라 암호문  + 평문도 암호문으로 인식
# bias 도 암호처리 해야 발표할때 뭔가 서버측에서도 weight 랑 bias를 암호화해서 관리하고 있다 보여줄수있을듯 