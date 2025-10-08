# =========================
# Streamlit Frontend (Online Calculator)
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

# Feature names (must match the training data column names and order)
feature_names = [
    "Age",                  # continuous (years)
    "RBC",                  # continuous (10^12/L)
    "FIGO",                 # categorical (encoded as below)
    "Transverse_Diameter",  # continuous (mm)
    "Pelvic_Invasion",      # categorical (Negative/Positive)
    "Radiotherapy",         # categorical (4 levels)
    "LNM"                   # categorical (Negative/Positive)
]

st.title("Cervical Adenocarcinoma — Survival Risk Calculator (Random Forest)")
st.markdown("""
*Enter seven variables (Age, RBC, FIGO stage, tumor transverse diameter, pelvic wall invasion, radiotherapy type, and lymph node metastasis) to obtain the model prediction and a SHAP-based single-case interpretation.  
For research use only; not for clinical decision-making.*
""")

# ---------- Categorical mappings (MUST match the encodings used during training) ----------
FIGO_map = {
    "Carcinoma in situ": 0,
    "Stage I": 1,
    "Stage II": 2,
    "Stage III": 3,
    "Stage IV": 4
}
Pelvic_map = {"Negative": 0, "Positive": 1}
RT_map = {
    "No radiotherapy": 0,
    "Definitive radiotherapy": 1,
    "Postoperative radiotherapy": 2,
    "Preoperative brachytherapy": 3
}
LNM_map = {"Negative": 0, "Positive": 1}

# ---------- Continuous inputs ----------
Age = st.number_input("Age (years):", min_value=24, max_value=80, value=49, step=1)
RBC = st.number_input("RBC (10^12/L):", min_value=2.0, max_value=7.0, value=4.25, step=0.1, format="%.1f")
Transverse_Diameter = st.number_input("Tumor transverse diameter (mm):", min_value=0.0, max_value=200.0, value=50.0, step=1.0, format="%.1f")

# ---------- Categorical inputs ----------
FIGO_label = st.selectbox("FIGO stage:", options=list(FIGO_map.keys()), index=1)
Pelvic_label = st.selectbox("Pelvic wall invasion:", options=list(Pelvic_map.keys()), index=0)
RT_label = st.selectbox("Radiotherapy (RT):", options=list(RT_map.keys()), index=0)
LNM_label = st.selectbox("Lymph node metastasis (LNM):", options=list(LNM_map.keys()), index=0)

# ---------- Encode categoricals ----------
FIGO = FIGO_map[FIGO_label]
Pelvic_Invasion = Pelvic_map[Pelvic_label]
Radiotherapy = RT_map[RT_label]
LNM = LNM_map[LNM_label]

# ---------- Assemble single-sample DataFrame ----------
feature_values = [Age, RBC, FIGO, Transverse_Diameter, Pelvic_Invasion, Radiotherapy, LNM]
X_user = pd.DataFrame([feature_values], columns=feature_names)

# =========================
# Prediction + SHAP single-case force plot
# =========================
if st.button("Predict"):
    proba = model.predict_proba(X_user)[0]
    pred_class = int(np.argmax(proba))
    risk_prob = float(proba[1])  # assumes class "1" is the event/high-risk class

    st.subheader("Prediction")
    st.write(f"**Predicted Class (0 = Low risk, 1 = High risk):** {pred_class}")
    st.write(f"**Predicted Probability of Class 1:** {risk_prob:.3f}")

    if pred_class == 1:
        st.info(
            f"Model indicates higher risk (P(class=1) = {risk_prob*100:.1f}%). "
            "Consider integrating with clinical staging and treatment plans, and intensify follow-up when appropriate."
        )
    else:
        st.info(
            f"Model indicates relatively low risk (P(class=1) = {risk_prob*100:.1f}%). "
            "Routine follow-up is suggested while monitoring key risk factors (FIGO stage, pelvic invasion, LNM)."
        )

    # SHAP single-case explanation (RandomForest + TreeExplainer)
    try:
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer.shap_values(X_user)

        # 处理 expected_value
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_value = explainer.expected_value[1] if len(np.atleast_1d(explainer.expected_value)) > 1 \
                         else np.atleast_1d(explainer.expected_value)[0]
        else:
            base_value = explainer.expected_value

        # ✅ 取单个样本的一维向量
        if isinstance(shap_values, list):
            shap_values_pos = shap_values[1][0]
        else:
            shap_values_pos = shap_values[0]
        shap_values_pos = np.array(shap_values_pos).flatten()

        shap.force_plot(
            base_value,
            shap_values_pos,
            X_user.iloc[0],
            matplotlib=True, show=False
        )
        plt.tight_layout()
        plt.savefig("shap_force_rf_single.png", dpi=600, bbox_inches='tight')
        plt.close()

        st.subheader("SHAP Force Plot (single case)")
        st.image("shap_force_rf_single.png",
                 caption="Feature contributions toward the predicted probability (Class = 1)")

    except Exception as e:
        st.warning(f"SHAP plotting error: {e}")
