import pandas as pd
import numpy as np
import heaan
from joblib import load

# 1. 병합된 전처리 결과 불러오기
df = pd.read_csv("data/merged_clean.csv")

# 2. X, y 분리
X = df.drop(columns=["SEQN", "DIQ010"])
X = X.select_dtypes(include=['float64', 'int64'])  # 숫자형 컬럼만 유지
y = df["DIQ010"].astype(int)

# 3. 학습된 weight와 bias 불러오기
weights = np.load("data/lr_weights.npy").flatten().tolist()
bias = np.load("data/lr_bias.npy").item()

# 4. HE 환경 초기화
log_slots = 4
context = heaan.Context(log_slots)
sk = heaan.SecretKey(context)
pk = heaan.PublicKey(context, sk)

# 4-1. 저장된 StandardScaler 불러오기
scaler = load("data/scaler.pkl")

# 5. sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 6. 7명 샘플 반복 예측 (정답 비교 포함)
correct = 0
print("\n[예측 결과 요약]")
for i in range(7):
    x_raw = X.iloc[i].values.tolist()
    x_vec = scaler.transform([x_raw])[0].tolist()  # 정규화 적용
    true_label = y.iloc[i]
    seqn = df.iloc[i]["SEQN"]

    # Message 인코딩
    m_x = heaan.Message(log_slots)
    m_w = heaan.Message(log_slots)
    for j in range(len(x_vec)):
        m_x[j] = x_vec[j]
        m_w[j] = weights[j]

    # 암호화
    ctxt_x = heaan.Ciphertext(context)
    ctxt_w = heaan.Ciphertext(context)
    context.encrypt(m_x, pk, ctxt_x)
    context.encrypt(m_w, pk, ctxt_w)

    # 곱셈
    ctxt_res = heaan.Ciphertext(context)
    context.multiply(ctxt_x, ctxt_w, ctxt_res)

    # 복호화
    res = heaan.Message(log_slots)
    context.decrypt(ctxt_res, sk, res)

    # dot product + bias
    dot_product = sum(res[j] for j in range(len(x_vec)))
    dot_product += bias

    # sigmoid + 예측값
    prob = sigmoid(dot_product)
    pred = 1 if prob > 0.5 else 0
    is_correct = "✅" if pred == true_label else "❌"
    
    print(f"환자 {i} (SEQN: {int(seqn)}): 예측={pred} 확률={round(prob, 4)} | 실제={true_label} {is_correct}")

    if pred == true_label:
        correct += 1

# 7. 정확도 출력
print(f"\n총 {correct}/7명 정답 (정확도: {correct / 7:.2%})")

