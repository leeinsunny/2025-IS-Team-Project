import os
import sys
import base64
import json
import numpy as np
from flask import Flask, request, jsonify, Response, send_file
import tempfile
from model_utils import load_scaler, load_model_parameters
import os
os.environ["HEAAN_TYPE"] = "pi" # use pi for using pi-heaan,you can use this for other ipynb files to using pi-heaan
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of CPU threads to use for parallel regions
import heaan_stat
from heaan_stat import Block, Context
# ───────────────────────────────────────────────────────────────────────────
# 프로젝트 루트로 이동 & 모듈 경로 추가 (is_back → tutorial)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
# ───────────────────────────────────────────────────────────────────────────



# HEaaN 환경 세팅
os.environ["HEAAN_TYPE"]      = "pi"
os.environ["OMP_NUM_THREADS"] = "8"

app = Flask(__name__)

# 필요한 디렉토리 자동 생성
# 디렉토리 준비
UPLOAD_BASE = "/root/2025-IS-Team-Project/is_back/uploads"
PK_DIR = os.path.join(UPLOAD_BASE, "public_keypack/PK")
os.makedirs(PK_DIR, exist_ok=True)

XHE_DIR = "/root/2025-IS-Team-Project/is_back/uploads/xhe"
os.makedirs(XHE_DIR, exist_ok=True)


# weight pre-storage
def encode_weight(context, w, b) -> Block:
    w_list = w.tolist()
    w_list.append(b)
    w_he = Block(context, data = w_list, encrypted=False)
    return w_he

#2. dot product calculation function with ciphertext
def encrypted_dot(x_he, w_he):
    prod_tmp = x_he * w_he # using level 1 
    # 3) Slot-by-slot multiplication and sum (ciphertext)
    prod = prod_tmp.sum(axis=1, direction=0, unit_shape=(16, 512))
    print('sum = ',prod)
    # Bias is included even if don't add it separately
    return prod

def load_x_he(context, x_he_path) -> Block:
    return Block(context,encrypted=True).load(x_he_path)

#연산을 위한 context생성
def make_context(key_dir):
    return Context( key_dir_path="/root/2025-IS-Team-Project/is_back/uploads", parameter = 'SS7', generate_keys=False,load_keys = 'pk')

#시그모이드 근사식
def encrypted_sigmoid_approximate_equation(z):
    #z level 6
    z2 = z * z #z2 level 5
    z3 = z * z2 #level 6 * level5 = level 4
    z5 = z3 * z2 #level 4 * level 5 = level 3
    z7 = z5 * z2 #level 3 * level 5 = level 2
    return 0.5 + 0.2166 * z - 0.0087 * z3 + 0.00023 * z5 - 0.0000021 * z7 #float * level 2 = level 1



def get_save_dir(filename):
    if filename.lower() == "secretkey.bin":
        return SECRET_DIR
    else:
        return PK_DIR

# 1) 암호문 전용: 근사 시그모이드만 적용
@app.route("/encrypted_sigmoid", methods=["POST"])
def sigmoid_endpoint():
    try:
        print('수신완료')

        # 2. Context 초기화 (프론트에서 넘어온 키로 생성)
        context = make_context("/root/2025-IS-Team-Project/is_back/uploads")
        print(context)
        print('컨텍스트 생성 완료')
        # 2. 프론트에서 넘어온 x_he.bin 파일 경로
        bin_path = os.path.join("/root/2025-IS-Team-Project/is_back/uploads", "xhe", "x_he.bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"{bin_path} 파일이 존재하지 않습니다.")
        
        # Block 객체로 로드
        x_he = load_x_he(context, bin_path)

        # 가중치, 편향 load
        weights, bias = load_model_parameters(
        w_path="data/lr_weights.npy",
        b_path="data/lr_bias.npy"
        )
        #가중치와 편향 연산을위해 block으로 encoding
        w_he = encode_weight(context, weights, bias)        
        print("✅ 모델 파라미터 부호화 완료")

        print(x_he)
        print(w_he)
        # 내적수행 
        dot_res = encrypted_dot(x_he, w_he)

        #내적값 시그모이드 근사식
        result_block = encrypted_sigmoid_approximate_equation(dot_res)
        print("✅ sigmoid 근사 완료")
        
        print(result_block)
        #시그모이드 근사식 결과값을 bin 파일로 저장
        result_path = "/root/2025-IS-Team-Project/is_back/result_block.bin"
        result_block.save(result_path)

        return send_file(
        result_path,
        as_attachment=True,
        download_name="result_block.bin",
        mimetype="application/octet-stream"
    )

    except Exception as e:
        print(f"❌ 처리 실패: {e}")
        return f"Error: {str(e)}", 500




#청크로 나눠서 전송
@app.route("/upload_chunk", methods=["POST"])
def upload_chunk():
    try:
        filename = request.form["filename"]
        index = int(request.form["index"])
        file = request.files["chunk"]

        save_dir = get_save_dir(filename)
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f"{filename}.part{index}")
        file.save(file_path)

        print(f"✅ 청크 저장: {file_path}")
        return "ok", 200

    except Exception as e:
        print(f"❌ upload_chunk 오류: {e}")
        return f"Error: {str(e)}", 500


@app.route("/upload_chunk/complete", methods=["POST"])
def complete_upload():
    # 저장 위치 결정
    try:
        filename = request.form["filename"]
        total_chunks = int(request.form["total_chunks"])

        save_dir = get_save_dir(filename)
        final_path = os.path.join(save_dir, filename)

        with open(final_path, "wb") as outfile:
            for i in range(total_chunks):
                part_path = os.path.join(save_dir, f"{filename}.part{i}")
                with open(part_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(part_path)

        print(f"✅ 조합 완료: {final_path}")
        return "File assembled", 200
    except Exception as e:
        print(f"❌ 조합 오류: {e}")
        return f"Error: {str(e)}", 500




@app.route("/upload_xhe_chunk", methods=["POST"])
def upload_xhe_chunk():
    try:
        filename = request.form["filename"]
        index = int(request.form["index"])
        file = request.files["chunk"]

        file_path = os.path.join(XHE_DIR, f"{filename}.part{index}")
        file.save(file_path)

        print(f"📥 x_he 청크 저장 완료: {file_path}")
        return "ok", 200
    except Exception as e:
        print(f"❌ upload_xhe_chunk 오류: {e}")
        return f"Error: {str(e)}", 500


@app.route("/upload_xhe_complete", methods=["POST"])
def complete_xhe_upload():
    try:
        filename = request.form["filename"]
        total_chunks = int(request.form["total_chunks"])

        final_path = os.path.join(XHE_DIR, filename)

        with open(final_path, "wb") as outfile:
            for i in range(total_chunks):
                part_path = os.path.join(XHE_DIR, f"{filename}.part{i}")
                with open(part_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(part_path)

        print(f"✅ x_he 조합 완료: {final_path}")
        return "x_he.bin assembled", 200
    except Exception as e:
        print(f"❌ complete_xhe_upload 오류: {e}")
        return f"Error: {str(e)}", 500


#프론트로 결과bin파일 저장해주는 api
@app.route("/download_result_block", methods=["GET"])
def download_result_block():
    result_path = "/root/2025-IS-Team-Project/is_back/result_block.bin"
    
    def generate():
        with open(result_path, "rb") as f:
            while True:
                chunk = f.read(4096)  # 4KB 청크
                if not chunk:
                    break
                yield chunk
    
    return Response(
        generate(),
        mimetype="application/octet-stream",
        headers={
            "Content-Disposition": "attachment; filename=result_block.bin"
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
