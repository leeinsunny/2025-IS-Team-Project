#📄 pages/1_📛_Health_Form.py
import streamlit as st
import numpy as np
import requests
import json
import  io
import base64
from joblib import load
import tempfile
import os
os.environ["HEAAN_TYPE"] = "pi" # use pi for using pi-heaan,you can use this for other ipynb files to using pi-heaan
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions
from heaan_stat import Block, Context
with open("style/custom_style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🩺 Enter Your Health Information")

# 기본 정보
name = st.text_input("Name", max_chars=50)
age = st.number_input("Age", min_value=0, step=1)
gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
phone = st.text_input("Phone Number (10–15 digits)", max_chars=15)

st.markdown("---")
st.markdown("### Required Health Indicators (All 10 items)")

# 건강 정보 입력
bmi        = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, step=0.1)
bp_sys     = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=250, step=1)
bp_dia     = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=30, max_value=150, step=1)
age_input  = age
gender_bin = 1.0 if gender.lower() == "male" else 0.0
smoker_bin = st.radio("Do you smoke?", ["Yes", "No"]) == "Yes"
smoker_bin = 1.0 if smoker_bin else 0.0
glucose    = st.number_input("Blood Glucose (mg/dL)", min_value=30.0, max_value=300.0, step=0.1)
insulin    = st.number_input("Insulin Level (uIU/mL)", min_value=1.0, max_value=200.0, step=0.1)
chol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=50.0, max_value=400.0, step=1.0)
sugar_intk = st.number_input("Daily Sugar Intake (g/day)", min_value=0.0, max_value=500.0, step=1.0)


# key generation
def make_keys(key_path):
    return Context(key_dir_path=key_path, parameter = 'SS7', generate_keys=True)

#encrypt
def encrypt_input(context, x):
    x.append(1) # add 1 for bias
    x_he = Block(context, data = x, encrypted=True)
    # reason why 6 level
    # dot mult 1 time, sigmoid mult 5 time / 1+5
    x_he.level_down(6) # Since the required mult operation is 6, the ciphertext size was reduced to level 6.
    return x_he


#서버에서 넘어오는 암호문bin을 load
def load_x_he(context, x_he_path) -> Block:
    return Block(context,encrypted=True).load(x_he_path)

# pk 키 서버로 전송
def send_keys(server_url, key_dir_path,chunk_size=1024 * 1024):
    """
    public_keypack 모든 .bin 키 파일을 청크로 서버에 전송
    (secret_keypack 제외)
    """
    pubkey_dir = os.path.join(key_dir_path, "public_keypack/PK")

    for filename in os.listdir(pubkey_dir):
        if not filename.endswith(".bin") :  
            continue

        file_path = os.path.join(pubkey_dir, filename)
        print(f"📤 전송 시작: {filename}")

        # 1. 청크 단위로 전송
        with open(file_path, "rb") as f:
            chunk_index = 0
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                files = {
                    'chunk': (f"{filename}.part{chunk_index}", chunk)
                }
                data = {
                    'filename': filename,
                    'index': str(chunk_index)
                }

                response = requests.post(f"{server_url}/upload_chunk", files=files, data=data)
                if response.status_code != 200:
                    print(f"❌ 청크 전송 실패: {filename}.part{chunk_index}, 상태코드: {response.status_code}")
                    return
                print(f"✅ 청크 전송 완료: {filename}.part{chunk_index}")
                chunk_index += 1

        # 2. 조합 요청
        response = requests.post(f"{server_url}/upload_chunk/complete", data={
            "filename": filename,
            "total_chunks": chunk_index
        })

        if response.status_code == 200:
            print(f"✅ 파일 조합 완료: {filename}")
        else:
            print(f"❌ 조합 실패: {filename}, 상태코드: {response.status_code}")



def send_xhe_file_in_chunks(file_path, server_url, chunk_size=1024 * 1024):
    """
    x_he.bin을 새로운 API로 전송 (청크 분할 포함)
    :param file_path: x_he.bin 파일 경로
    :param server_url: 서버 주소 (http://127.0.0.1:5000)
    :param chunk_size: 1MB 기본
    """
    filename = os.path.basename(file_path)
    total_chunks = (os.path.getsize(file_path) + chunk_size - 1) // chunk_size

    with open(file_path, "rb") as f:
        for i in range(total_chunks):
            chunk = f.read(chunk_size)
            files = {"chunk": (f"{filename}.part{i}", chunk)}
            data = {"filename": filename, "index": str(i)}

            response = requests.post(f"{server_url}/upload_xhe_chunk", files=files, data=data)
            if response.status_code == 200:
                print(f"✅ x_he 청크 전송 완료: {filename}.part{i}")
            else:
                print(f"❌ x_he 청크 전송 실패: {filename}.part{i}, 상태코드: {response.status_code}")
                return

    # 조합 요청
    response = requests.post(f"{server_url}/upload_xhe_complete", data={
        "filename": filename,
        "total_chunks": total_chunks
    })

    if response.status_code == 200:
        print(f"✅ x_he 조합 완료: {filename}")
    else:
        print(f"❌ 조합 실패: {response.status_code} - {response.text}")

#시그모이드식 수행 api
def call_encrypted_sigmoid_api():
    server_url = "http://127.0.0.1:5000/encrypted_sigmoid"
    try:
        response = requests.post(server_url)

        if response.status_code == 200:
            print("✅ sigmoid 연산 성공")
            
            result = response.content  
            print("📩 결과:", result)
        else:
            print(f"❌ 오류 발생: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ 요청 실패: {e}")



# ✅ Submit
if st.button("Submit"):
    if not name or not gender or not phone or len(phone) < 10:
        st.warning("Please fill in all required fields correctly.")
    else:
        st.success("✅ Form submitted!")
        #사용자 인풋값
        x_raw = [
            bmi, bp_sys, bp_dia, age_input,
            gender_bin, smoker_bin,
            glucose, insulin, chol_total, sugar_intk
        ]

        st.session_state.user_data = {
            "name": name,
            "age": age,
            "gender": gender,
            "phone": phone,
            "inputs": x_raw
        }

        #  Normalize
        scaler = load("data/scaler.pkl")
        x_scaled = scaler.transform([x_raw])[0].tolist()

        #  Encrypt with HEaaN
        context = make_keys("./key_path")
        x_he = encrypt_input(context, x_scaled)
        print(x_he)
        # 만들어진 키를 서버에 전송
        send_keys("http://127.0.0.1:5000", "./key_path")
        
      
        # 암호화된 사용자 input값을 bin 파일로 저장
        x_he.save("./x_he.bin")
        # 이 파일을 서버에 전송
        send_xhe_file_in_chunks("./x_he.bin", "http://127.0.0.1:5000")
        
        #시그모이드 시행 함수 호출-> 서버에서 저장된 암호문bin 파일과 pk파일로 연산 수행후 나온 결과값 bin 파일로 저장되어있음.
        ct_after = call_encrypted_sigmoid_api()

        #서버에 저장된 연산 결과값bin 파일을 get 
        response = requests.get("http://127.0.0.1:5000/download_result_block", stream=True)
        with open("data/result_block.bin", "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
        print("✅ result_block.bin 저장 완료")

        # 저장한 연산결과값bin 파일을 load
        ct_result = load_x_he(context,'/root/2025-IS-Team-Project/is_front/data/result_block.bin')
        
        print(ct_result)
        # block 형식 암호문 복호화

        prob = ct_result.decrypt().to_series().iloc[0]
        print(prob)
        # 당뇨 예측 
        pred = int(prob >= 0.5)

            # 🔹 Save result for UI
        result = {"confidence": float(prob), "prediction": pred}
        with open("data/result.json", "w") as f:
            json.dump(result, f)

        st.success("✅ Prediction completed! Redirecting...")
        st.switch_page("pages/Prediction_Result.py")

