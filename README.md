# 📊 Privacy‐Preserving Diabetes Risk Prediction

> **팀 프로젝트 목표**
1. **로지스틱 회귀 모델 학습**  
   - NHANES 2013–2014 건강 데이터를 기반으로 주요 피처(공복 혈당, BMI, 인슐린 수치, 혈압 등)를 활용해  
   - 당뇨병 발병 확률을 0–1 범위의 확률값으로 예측하는 모델 구축 :contentReference[oaicite:0]{index=0}

2. **CKKS 동형암호 기반 안전 추론 파이프라인 구현**  
   - HeaAN Stat 시뮬레이터를 사용한 CKKS 암호화  
   - 암호화된 상태에서 평균, 상관분석, 로지스틱 회귀 추론 등 연산 수행  
   - 데이터 복호화 없이 예측 결과 제공 :contentReference[oaicite:1]{index=1}
---

## 🚩 목차

1️⃣ [진행 현황 & 체크리스트](#1️⃣-진행-현황--체크리스트)
2️⃣ [프로젝트 구조](#2️⃣-프로젝트-구조)
3️⃣ [환경 설정](#3️⃣-환경-설정)
4️⃣ [사용법](#4️⃣-사용법)
5️⃣ [향후 계획](#5️⃣-향후-계획)

---

## 1️⃣ 진행 현황 & 체크리스트

|  단계 | 설명                                                                                                                                        |    상태   |
| :-: | :---------------------------------------------------------------------------------------------------------------------------------------- | :-----: |
| 1️⃣ | **데이터 준비**<br>- NHANES CSV (`demographic.csv`, `diet.csv`, `examination.csv`, `labs.csv`, `questionnaire.csv`)<br>- `merged_clean.csv` 생성 |   ✅ 완료  |
| 2️⃣ | **평문 모델 학습** (`train_model.py`)<br>- StandardScaler 학습 → `scaler.pkl` 저장<br>- LogisticRegression 학습 → `lr_weights.npy`, `lr_bias.npy` 저장  |   ✅ 완료  |
| 3️⃣ | **HEaaN Stat 시뮬 시연** (`*.ipynb`)<br>- `HESeries`, `HEFrame` 기본 연산<br>- 벡터 곱셈/덧셈, 통계 기능                                                    |   ✅ 완료  |
| 4️⃣ | **모듈화**<br>- `model_utils.py`<br>- `he_ops.py`                                                                                            |   ✅ 완료  |
| 5️⃣ | **암호화된 추론** (`predict_encrypted.py`)<br>- 사용자 입력 → 암호화<br>- Homomorphic 곱셈 → 복호화 → sigmoid → 예측 결과 저장                                     |   ✅ 완료  |
| 6️⃣ | **UI/시각화**<br>- 사용자 프론트엔드 화면 설계 <br>- 상관관계 히트맵                                                                                              | 🔲 진행 중 |

---
## 2️⃣ 프로젝트 구조

- **TUTORIAL/**  
  - **data/** ← 원본·전처리된 NHANES 데이터  
    - `demographic.csv`  
    - `diet.csv`  
    - `examination.csv`  
    - `labs.csv`  
    - `questionnaire.csv`  
    - `merged_clean.csv` ← 통합·정제된 학습용 데이터  
    - `scaler.pkl` ← StandardScaler 객체  
    - `lr_weights.npy` ← 학습된 로지스틱 회귀 가중치  
    - `lr_bias.npy` ← 학습된 절편(bias)  
    - `prediction_result.csv` ← 예측 스크립트 결과 저장  
  - **keys/** ← pi-heaan 시뮬레이터 키 파일  
  - `train_model.py` ← (1️⃣) 평문 모델 학습 스크립트  
  - `model_utils.py` ← scaler·파라미터 로더  
  - `he_ops.py` ← Context 생성·암호화 내적 함수  
  - `predict_encrypted.py` ← (5️⃣) 암호화된 추론 파이프라인  
  - `new_predict_encrypted.py` ← 모듈화 버전  
  - `*.ipynb` ← HEaaN.Stat 기본 실습 노트북  

---
## 3️⃣ 환경 설정

1. **Docker & WSL2**

   * Docker Desktop 설치 + WSL2 통합 활성화
2. **HEaaN Stat 컨테이너 실행**

   ```bash
   docker run -p 8888:8888 -it cryptolabinc/heaan-stat:1.0.0-cpu
   ```
3. **VSCode → Remote Explorer → Attach to Container**
4. **컨테이너 내부에서 라이브러리 설치**

   ```bash
   pip install scikit-learn pandas numpy joblib streamlit
   ```
5. **코드 복사/동기화**

   * `docker cp` 혹은 VSCode drag\&drop 으로 `TUTORIAL/` 폴더 동기화

---

## 4️⃣ 사용법

### 4.1 평문 모델 학습

```bash
cd /root/tutorial
python3 train_model.py
```

### 4.2 암호화된 추론

```bash
python3 predict_encrypted.py
# 또는 모듈화 버전
python3 new_predict_encrypted.py
```

### 4.3 (옵션) Streamlit UI

```bash
streamlit run app.py
```

---

## 5️⃣ 향후 계획

* 🔳 **UI 고도화**: 사용자로부터 건강 정보를 입력받고, 입력받은 건강정보를 바탕으로 당뇨병 예측
* 🔳 **성능 평가**: 평문 vs 암호문 추론 속도/정확도 비교
* 🔳 **보고서 & 발표 자료**: 구현 구조도, 실험 결과, 보안/프라이버시 해설


