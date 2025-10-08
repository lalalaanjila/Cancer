# =========================
# Streamlit Frontend (Random Forest + SHAP waterfall)
# Run from terminal:  streamlit run <this_file>.py
# =========================
import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model(path='RandomForest.pkl'):
    return joblib.load(path)

model = load_model('RandomForest.pkl')

# Feature names (must match training order)
feature_names = [
    "Age", "RBC", "FIGO", "Transverse_Diameter",
    "Pelvic_Invasion", "Radiotherapy", "LNM"
]

st.title("Cervical Adenocarcinoma — Survival Risk Calculator (Random Forest)")
st.markdown("""
*Enter seven variables (Age, RBC, FIGO stage, tumor transverse diameter, pelvic wall invasion, radiotherapy type, and lymph node metastasis) to obtain the model prediction and a SHAP-based single-case interpretation.  
For research use only; not for clinical decision-making.*
""")

# ---------- Input ----------
Age = st.number_input("Age (years):", min_value=24, max_value=80, value=49, step=1)
RBC = st.number_input("RBC (10^12/L):", min_value=2.0, max_value=7.0, value=4.25, step=0.1, format="%.1f")
Transverse_Diameter = st.number_input("Tumor transverse diameter (mm):", min_value=0.0, max_value=200.0, value=50.0, step=1.0, format="%.1f")

FIGO_map = {"Carcinoma in situ": 0, "Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4}
Pelvic_map = {"Negative": 0, "Positive": 1}
RT_map = {"No radiotherapy": 0, "Definitive radiotherapy": 1, "Postoperative radiotherapy": 2, "Preoperative brachytherapy": 3}
LNM_map = {"Negative": 0, "Positive": 1}

FIGO_label = st.selectbox("FIGO stage:", options=list(FIGO_map.keys()), index=1)
Pelvic_label = st.selectbox("Pelvic wall invasion:", options=list(Pelvic_map.keys()), index=0)
RT_label = st.selectbox("Radiotherapy (RT):", options=list(RT_map.keys()), index=0)
LNM_label = st.selectbox("Lymph node metastasis (LNM):", options=list(LNM_map.keys()), index=0)

FIGO = FIGO_map[FIGO_label]
Pelvic_Invasion = Pelvic_map[Pelvic_label]
Radiotherapy = RT_map[RT_label]
LNM = LNM_map[LNM_label]

# ---------- Assemble DataFrame ----------
feature_values = [Age, RBC, FIGO, Transverse_Diameter, Pelvic_Invasion, Radiotherapy, LNM]
X_user_df = pd.DataFrame([feature_values], columns=feature_names)

# =========================
# Prediction + SHAP waterfall plot
# =========================
if st.button("Predict"):
    proba = model.predict_proba(X_user_df)[0]
    pred_class = int(np.argmax(proba))
    risk_prob = float(proba[1])

    st.subheader("Prediction")
    st.write(f"**Predicted Class (0 = Low risk, 1 = High risk):** {pred_class}")
    st.write(f"**Predicted Probability of Class 1:** {risk_prob:.3f}")

    if pred_class == 1:
        st.info(f"Model indicates higher risk (P(class=1) = {risk_prob*100:.1f}%). "
                "Consider integrating with clinical staging and treatment plans, and intensify follow-up when appropriate.")
    else:
        st.info(f"Model indicates relatively low risk (P(class=1) = {risk_prob*100:.1f}%). "
                "Routine follow-up is suggested while monitoring key risk factors (FIGO stage, pelvic invasion, LNM).")

    # SHAP explanation (waterfall)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_user_df)

        # 取类别1的 SHAP 值
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1][0]
            base_value = explainer.expected_value[1]
        else:
            shap_values_pos = shap_values[0]
            base_value = explainer.expected_value

        # 构造 Explanation 对象（新版 SHAP 用 base_values）
        shap_exp = shap.Explanation(
            values=shap_values_pos,
            base_values=base_value,
            data=X_user_df.iloc[0].values,
            feature_names=feature_names
        )

        # 绘制 waterfall 图
        shap.plots._waterfall.waterfall_legacy(shap_exp)
        plt.tight_layout()
        plt.savefig("shap_waterfall_rf_single.png", dpi=600, bbox_inches='tight')
        plt.close()

        st.subheader("SHAP Waterfall Plot (single case)")
        st.image("shap_waterfall_rf_single.png",
                 caption="Feature contributions toward predicted probability (Class = 1)")

    except Exception as e:
        st.warning(f"SHAP plotting error: {e}")
