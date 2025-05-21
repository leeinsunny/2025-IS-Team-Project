import numpy as np
import pandas as pd
from joblib import load
from heaan_stat import Context, HESeries

# 1. 사용자 입력값 (예시: BMI, 혈압 등)
x_raw = [23.1, 120.0, 75.0, 88.5, 5.0, 1.0, 0.0, 1.0, 0.0, 0.0]  # 입력 벡터 길이 = 10
print("✅ 사용자 입력값:", x_raw)

# 2. 스케일러 로드 및 정규화
scaler = load("data/scaler.pkl")
x_scaled = scaler.transform([x_raw])[0].tolist()
print("✅ 정규화 결과:", x_scaled)

# 3. 학습된 모델 파라미터 불러오기 (가중치, 편향)
weights = np.load("data/lr_weights.npy")
bias = np.load("data/lr_bias.npy").item()

# 4. HE 컨텍스트 초기화 (기존 키 사용)
context = Context(
    key_dir_path="./keys",
    generate_keys=False
)

# 5. 암호화된 HESeries(암호화된 벡터) 생성
x_he = HESeries(context, x_scaled)
w_he = HESeries(context, weights.tolist())

# 6. 암호문 상태에서 곱셈 수행
prod = x_he * w_he

# 7. 복호화 및 평문 벡터 반환
# prod는 아직 암호문 상태
prod_plain = prod.to_series()  # 내부에서 한 번만 decrypt 호출
print("1) 복호화된 곱셈 벡터:", prod_plain.tolist())

# 8. 평문에서 dot‐product 계산
dot_plain = prod_plain.sum()
print("2) dot product 합계:", dot_plain)

# 9. 시그모이드 적용 및 예측
z    = dot_plain + bias
prob = 1/(1+np.exp(-z))
pred = int(prob>=0.5)

print("3) 예측 확률:", round(prob,4))
print("4) 예측 결과 (0=No,1=Yes):", pred)

# 10. 결과 저장
result = pd.DataFrame({
    "input":       [x_raw],
    "probability": [round(prob,4)],
    "prediction":  [pred]
})
result.to_csv("data/prediction_result.csv", index=False)
print("5) 예측 결과 저장 완료: prediction_result.csv")
