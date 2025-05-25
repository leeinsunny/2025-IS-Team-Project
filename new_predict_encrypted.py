# 이건 predict_encrypted.py랑 동일한데, he_ops.py랑 model_utils.py 써서 모듈화한거
import numpy as np
import pandas as pd
import os
os.environ["HEAAN_TYPE"] = "pi" # use pi for using pi-heaan,you can use this for other ipynb files to using pi-heaan
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions
from model_utils import load_scaler, load_model_parameters
from he_ops      import make_context, encrypted_dot

import heaan_stat
# -- 1. 사용자 입력값 (예시: BMI, 혈압 등)
x_raw = [26.7, 122.0, 72.0, 69, 1.0, 1.0, 134.59, 14.94, 209.0, 176.47]
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


#평문내적계산
dot_plain = np.dot(weights, x_scaled)
print(f"✅  Dot product (평문): {dot_plain:.4f}")
print(bias)
# -- 6. 시그모이드 적용 및 예측
z  = dot_plain + bias
print('bias포함z값',z)

# pred = int(prob >= 0.5)
# print(f"✅ 예측 확률: {prob:.4f}")
# print(f"✅  예측 결과 (0=No, 1=Yes): {pred}")


def encrypted_sigmoid_approximate_equation(z):
    z1 = z
    z3 = z1 * z1 * z1
    z5 = z3 * z1 * z1
    z7 = z5 * z1 * z1
    return 0.5 + 0.2166 * z1 - 0.0087 * z3 + 0.00023 * z5 - 0.0000021 * z7

def encrypted_sigmoid(z):
    
    return 1 / (1 + np.exp(-z))

approximate_t = encrypted_sigmoid_approximate_equation(z)
t = encrypted_sigmoid(z)
print('근사식에 넣은값', approximate_t)
print('시그모이드식에에 넣은값',t)

# -- 7. 결과 저장
# result = pd.DataFrame({
#     "input":       [x_raw],
#     "probability": [round(prob, 4)],
#     "prediction":  [pred]
# })
# result.to_csv("data/prediction_result.csv", index=False)
# print("✅ 예측 결과 저장 완료: data/prediction_result.csv")
