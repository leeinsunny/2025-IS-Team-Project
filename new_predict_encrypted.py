# 이건 predict_encrypted.py랑 동일한데, he_ops.py랑 model_utils.py 써서 모듈화한거
import numpy as np
import pandas as pd

from model_utils import load_scaler, load_model_parameters
from he_ops      import make_context, encrypted_dot

# -- 1. 사용자 입력값 (예시: BMI, 혈압 등)
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

# -- 4. HE 컨텍스트 초기화 (기존 키 사용)
ctx = make_context(key_dir="./keys")

# -- 5. 암호화된 내적(dot product) 계산
dot_plain = encrypted_dot(ctx, x_scaled.tolist(), weights)
print(f"✅  Dot product (복호화된 실수): {dot_plain}")

# -- 6. 시그모이드 적용 및 예측
z    = dot_plain + bias
prob = 1 / (1 + np.exp(-z))
pred = int(prob >= 0.5)
print(f"✅ 예측 확률: {prob:.4f}")
print(f"✅  예측 결과 (0=No, 1=Yes): {pred}")

# -- 7. 결과 저장
result = pd.DataFrame({
    "input":       [x_raw],
    "probability": [round(prob, 4)],
    "prediction":  [pred]
})
result.to_csv("data/prediction_result.csv", index=False)
print("✅ 예측 결과 저장 완료: data/prediction_result.csv")
