import streamlit as st
import plotly.graph_objects as go
import json
import os
import random

# ⬇️ 경로 설정 (data/result.json 경로 계산)
base_dir = os.path.dirname(os.path.dirname(__file__))  # is_front/pages/ → is_front/
result_path = os.path.join(base_dir, "data", "result.json")

# 🎨 CSS 스타일 적용
with open("style/custom_style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Prediction Result")

# 🔐 사용자 정보 세션 확인
if "user_data" not in st.session_state or not st.session_state.user_data:
    st.warning("No data submitted. Please go back to the form.")
    st.stop()

data = st.session_state.user_data

# 🔎 사용자 정보 출력
st.markdown(f"""
**Name:** {data['name']}  
**Age:** {data['age']}  
**Gender:** {data['gender']}  
**Phone:** {data['phone']}  
""")

st.markdown("---")

# ✅ 암호화 추론 결과 불러오기
try:
    with open(result_path) as f:
        result = json.load(f)
    confidence = float(result["confidence"])
    prediction = "Diabetic" if result["prediction"] == 1 else "Non-Diabetic"
    if not (0 <= confidence <= 1):
        raise ValueError("Confidence out of bounds")
    confidence_pct = confidence * 100
except Exception as e:
    confidence = round(random.uniform(0.5, 0.95), 2)
    confidence_pct = confidence * 100
    prediction = "Diabetic" if confidence > 0.7 else "Non-Diabetic"
    st.warning(f"⚠️ 암호화 추론 결과 파일 오류로 랜덤 예측을 사용합니다. ({e})")

st.markdown(f"### 🔐 Predicting diabetes risk using encrypted AI model... ({confidence_pct:.2f}%)")

# ✅ 예측 결과 시각화 (도넛 그래프)
labels = ["Diabetic", "Non-Diabetic"]
values = [confidence_pct, 100 - confidence_pct]
colors = ["#ff6a00", "#008cff"]

fig1 = go.Figure(data=[
    go.Pie(labels=labels, values=values, hole=0.6, marker=dict(colors=colors))
])
fig1.update_layout(
    title_text="Diabetes Risk Confidence",
    annotations=[dict(text=f"{confidence_pct:.2f}%", x=0.5, y=0.5, font_size=20, showarrow=False)]
)
st.plotly_chart(fig1, use_container_width=True)

# ✅ 입력값 vs 정상 범위 비교
st.markdown("### 📏 Deviation from Normal Ranges")

# 🔁 리스트 입력값일 경우 dict로 변환
input_keys = [
    "bmi", "bp_sys", "bp_dia", "age",
    "gender_bin", "smoker_bin",
    "glucose", "insulin", "chol_total", "sugar_intake"
]

input_metrics = data.get("inputs", {})
if isinstance(input_metrics, list):
    input_metrics = dict(zip(input_keys, input_metrics))

if not isinstance(input_metrics, dict):
    st.error("❌ Invalid input format: expected a dictionary or list.")
    st.stop()

flat = {}
for k, v in input_metrics.items():
    try:
        flat[k] = float(v)
    except:
        continue

# 정상 범위 기준 설정
normal_ranges = {
    "bmi": (18.5, 24.9),
    "bp_sys": (90, 120),
    "bp_dia": (60, 80),
    "insulin": (2.0, 25.0),
    "chol_total": (0, 200),
    "sugar_intake": (0, 50),
    "glucose": (70, 99),
}

bars = []
lines = []
for key, value in flat.items():
    for norm_key in normal_ranges:
        if norm_key in key:
            low, high = normal_ranges[norm_key]
            bars.append((key, value, f"{low}-{high}"))
            lines.append((key, low, high))

# 📊 바 차트 시각화
if bars:
    x = [b[0] for b in bars]
    y = [b[1] for b in bars]

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=x, y=y, name="Your Value", marker_color="#008cff"))

    for i, (k, low, high) in enumerate(lines):
        fig2.add_shape(type="rect", x0=i-0.4, x1=i+0.4, y0=low, y1=high,
                       fillcolor="LightGreen", opacity=0.3, line_width=0)

    fig2.update_layout(title="Comparison with Normal Ranges",
                       yaxis_title="Value", xaxis_title="Metric")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No matching numerical inputs to compare with normal ranges.")

# ✅ 요약 카드
st.markdown("### 📝 Summary")
color = "red" if prediction == "Diabetic" else "green"
st.markdown(f"""
- **Prediction:** <span style='color:{color}'><strong>{prediction}</strong></span>  
- **Confidence:** {confidence_pct:.2f}%
""", unsafe_allow_html=True)

st.markdown("---")
st.button("🔁 Start Over", on_click=lambda: st.session_state.clear())
