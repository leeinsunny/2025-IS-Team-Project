'''
1️⃣ 평문 모델 학습
- 파일: train_model.py
1) data/merged_clean.csv 로드
2) 특성(X)·레이블(y) 분리 → 숫자형만 선택
3) StandardScaler 학습 → scaler.pkl 저장
4) LogisticRegression 학습 → coef_(weights), intercept_(bias) 저장 (lr_weights.npy, lr_bias.npy)
5) 가중치·바이어스 shape 확인
6) data 폴더에 scaler.pkl, lr_weights.npy, lr_bias.npy 저장
7) 학습 완료 메시지 출력
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump
import os

# 1. 데이터 불러오기
df = pd.read_csv("data/merged_clean.csv")
print("✅ 데이터 로드 완료:", df.shape)

# 2. X, y 분리
X = df.drop(columns=["SEQN", "DIQ010"])
X = X.select_dtypes(include=["float64", "int64"])  # 숫자형만
y = df["DIQ010"].astype(int)

# 3. 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 모델 학습
model = LogisticRegression()
model.fit(X_scaled, y)

# 5. 가중치 및 바이어스 추출
weights = model.coef_[0]
bias = model.intercept_[0]

print("✅ 학습 완료")
print("가중치 shape:", weights.shape)
print("바이어스:", bias)

# 6. 저장 (모두 data 폴더에)
os.makedirs("data", exist_ok=True)
np.save("data/lr_weights.npy", weights)
np.save("data/lr_bias.npy", np.array([bias]))
dump(scaler, "data/scaler.pkl")

# 7. 모델 저장
print("✅ 저장 완료 (weights, bias, scaler)")

