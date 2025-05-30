# 2025 Team 8 – Privacy-Preserving Diabetes Prediction System

## 🔍 Project Overview

This project implements a fully encrypted diabetes prediction system based on CKKS homomorphic encryption (HE), using the NHANES 2013–2014 dataset. Users' health information is encrypted on the client side and processed securely on the server without any decryption, ensuring complete data privacy.

## 📁 System Components

- **Frontend**: Streamlit interface, CKKS key generation and encryption using HEaaN
- **Backend**: Flask API, inference with pre-trained logistic regression model
- **Encryption Library**: HEaaN Stat SDK (CKKS scheme)
- **Model**: Logistic Regression (`class_weight='balanced'`), AUC ≥ 0.8

---

## ⚙️ 1. Execution Manual

### 0️⃣ Prerequisites

- Python ≥ 3.10
- Docker (for HEaaN container) strongly recommended
- Ubuntu 20.04+ or WSL2 environment

### 1️⃣ Backend (Flask)

```bash
cd 2025-IS-Team-Project/is_back
pip install -r requirements.txt
python3 app.py
```

- Creates the following folders:
  - `is_back/uploads/public_keypack/`
  - `is_back/uploads/xhe/`
- **Backend must be started before the frontend.**

### 2️⃣ Frontend (Streamlit)

(Open a new terminal window)

```bash
cd 2025-IS-Team-Project/is_front
pip install -r requirements.txt
streamlit run streamlit_app.py
```

- Visit: [http://localhost:8501](http://localhost:8501)

---

## 👤 2. User Input Example

| Field               | Description                    | Example Value |
|--------------------|--------------------------------|---------------|
| Name               | 사용자 이름                     | John Doe      |
| Age                | 나이                            | 62            |
| Gender             | 성별 (1=Male, 0=Female)         | 1.0           |
| BMI                | 체질량지수                      | 25.8          |
| Systolic BP        | 수축기 혈압                      | 134.0         |
| Diastolic BP       | 이완기 혈압                      | 64.0          |
| Insulin            | 인슐린 수치 (uU/mL)              | 14.94         |
| Glucose (2hr OGTT) | 2시간 경구당부하 혈당             | 134.60        |
| Cholesterol        | 총 콜레스테롤 (mg/dL)            | 131.0         |
| Sugar Intake       | 당 섭취량 (g/day)                | 41.09         |
| Smoking Status     | 흡연 여부 (1=Yes, 0=No)          | 1.0           |

---

## 🔐 3. Encryption Flow (Client: `Health_Form.py`)

1. Normalize input and append `1.0` for bias term
2. Generate HE keys (`public`, `secret`, `relin`, `rotate`) → stored in `is_front/keys/`
3. Encrypt normalized input → store as `x_he.bin`
4. Send `x_he.bin` + `public_keypack` to:
   - `is_back/uploads/xhe/`
   - `is_back/uploads/public_keypack/`
5. Trigger prediction via `/predict` endpoint

---

## ⚙️ 4. Server-side Flow (Backend)

1. Load `public_keypack` to reconstruct HE context
2. Load `x_he.bin` → restore ciphertext input
3. Load model weights + bias and encode as plaintext block
4. Perform dot product → add bias (merged into input)
5. Apply polynomial sigmoid approximation
6. Store result ciphertext → returned to client

---

## 🔓 5. Decryption and Visualization (Client)

- Load `data/result_block.bin`
- Decrypt using secret key from local `keys/`
- Visualize prediction using Streamlit

---

## 🗂 6. Directory Structure Summary

| File/Key                 | Path                            | Description                      |
|--------------------------|---------------------------------|----------------------------------|
| Public KeyPack           | `is_back/uploads/public_keypack`| Server-side context for HE       |
| Encrypted Input (`x_he`) | `is_back/uploads/xhe/x_he.bin`  | User ciphertext                  |
| Encrypted Output         | `is_front/data/result_block.bin`| Encrypted prediction             |
| Secret Key + All Keys    | `is_front/keys/`                | Stored locally on client         |

---

## 🎯 Final Notes

- The **entire pipeline** runs with encrypted data.
- No plaintext is stored or processed on the server.
- Decryption is strictly local (client-side).
- The project is end-to-end HE compliant and privacy-preserving.

---

## 📹 Demonstration Video

- [Google Drive Link](https://drive.google.com/file/d/1ln6-8D9kLNkrq_-0MTEetzu5VM0e-Xzf/view?usp=drive_link)

---

## 📚 References

- NHANES 2013–2014 Dataset (CDC)
- HEaaN GitHub: https://github.com/snucrypto/HEaan
- American Diabetes Association Guidelines

---

## 📄 Project Documents

| Type       | Description                                                    | Download Link                                                                 |
|------------|----------------------------------------------------------------|--------------------------------------------------------------------------------|
| 📝 Proposal  | Initial planning, NHANES data selection, encryption model design | [IS_Proposal_Team8.pdf](./IS%20Proposal%20Team8.pdf)                           |
| 🔧 Progress  | Model performance comparison, recall-priority strategy        | [IS_Progress_Team8.pdf](./IS%20Progress%20Team8.pdf)                           |
| 📊 Final     | Scenario-driven HE pipeline design + full encryption workflow  | [IS_Final_Team8.pptx.pdf](./IS%20Final%20Team8.pptx.pdf)                       |

> These documents cover the full lifecycle of the project, from idea to final implementation using homomorphic encryption for secure diabetes prediction.

