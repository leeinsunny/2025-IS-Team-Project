# 📁 streamlit_app.py
import streamlit as st
from PIL import Image
import os
os.environ["HEAAN_TYPE"]      = "pi"
os.environ["OMP_NUM_THREADS"] = "8"

# Streamlit app settings
st.set_page_config(page_title="Secure Healthcare AI", layout="centered")

with open("style/custom_style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ✅ Hero section HTML only (style comes from custom_style.css)
st.markdown("""
<div class="hero">
    <h1>Lead the Future of Healthcare:<br>Secure, Smart, and Ethical.</h1>
    <p>Predict diabetes risk with encrypted AI.<br>Protect privacy, empower progress.</p>
</div>
""", unsafe_allow_html=True)

# 📄 Plain content (not center-aligned)
st.markdown("""
---

### 🧬 Project Overview

This system is built upon the NHANES dataset from the U.S. CDC and predicts an individual's **risk of diabetes** using input health indicators such as BMI, blood pressure, insulin levels, and more.

🔐 **CKKS homomorphic encryption** ensures that all personal health data remains encrypted throughout the process—never decrypted.

🔍 The prediction results are visualized in a user-friendly dashboard including:

- 📈 **Donut chart for diabetes risk**
- 📏 **Input values vs. standard reference ranges**
- 🔬 **Correlation heatmap between selected health metrics**

---

### 🧑‍💻 How to Use

1. Click the `Let's Get Started` button below  
2. Enter your health data by selecting and filling in relevant items  
3. Review the prediction and data visualizations  

All input data is processed securely in memory and is never saved. Feel confident and protected 💙🧡

---

<div style='text-align:center;'>
    <a href='/Health_Form' target='_self'>
        <button style="padding:12px 25px;font-size:16px;background-color:#ff6a00;color:white;border:none;border-radius:6px;">
            Let's Get Started 🚀
        </button>
    </a>
</div>
""", unsafe_allow_html=True)
