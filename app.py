import json
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
import librosa
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Multimodal Chest Pain & Asthma Triage", layout="wide")

# --- LOAD MODELS ONCE (CACHED) ---
@st.cache_resource
def load_pipeline():
    m1 = xgb.XGBClassifier()
    m1.load_model("model_stage1.json")
    
    m2 = xgb.XGBClassifier()
    m2.load_model("model_stage2.json")
    
    cnn = tf.keras.models.load_model("cnn_audio_model.keras")
    explainer = shap.TreeExplainer(m2)
    
    with open("config.json") as f:
        thresh = json.load(f)["best_thresh"]
    return m1, m2, cnn, explainer, thresh

model_stage1, model_stage2, cnn_audio_model, explainer_stage2, best_thresh = load_pipeline()

SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def process_audio(uploaded_file):
    y, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE, duration=DURATION)
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return spec_norm[..., np.newaxis]

# --- UI LAYOUT ---
st.title("Multimodal Chest Pain & Respiratory Triage System")
st.caption("Educational & Research Prototype — Not a substitute for professional medical diagnosis.")

# CYCLE 1
st.header("Cycle 1: Emergency Cardiac Rule-Out")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 90, 34)
    hr = st.slider("Resting Heart Rate (BPM)", 50, 160, 78)
with col2:
    rad = st.checkbox("Pain radiates to left arm, neck, or jaw")
    sob = st.checkbox("Severe shortness of breath at rest")
    sweat = st.checkbox("Cold sweats or dizziness")
    ecg = st.checkbox("Abnormal ECG reading (if known)")

if "cycle1_passed" not in st.session_state:
    st.session_state.cycle1_passed = False

if st.button("Run Cycle 1: Cardiac Triage", type="primary"):
    df1 = pd.DataFrame([{
        'age': age, 'heart_rate': hr, 'radiating_pain': int(rad),
        'shortness_of_breath': int(sob), 'sweating': int(sweat), 'abnormal_ecg': int(ecg)
    }])
    cardiac_prob = float(model_stage1.predict_proba(df1))
    if cardiac_prob >= 0.45:
        st.session_state.cycle1_passed = False
        st.error(f"⚠️ HIGH CARDIAC RISK DETECTED ({cardiac_prob*100:.1f}%). Urgent medical evaluation required.")
    else:
        st.session_state.cycle1_passed = True
        st.success(f"✅ Cycle 1 Passed: Low Cardiac Risk ({cardiac_prob*100:.1f}%). Cycle 2 unlocked below.")

# CYCLE 2
if st.session_state.cycle1_passed:
    st.divider()
    st.header("Cycle 2: Musculoskeletal & Respiratory Multimodal Assessment")
    
    c1, c2 = st.columns(2)
    with c1:
        tender = st.checkbox("Pressing directly on rib-breastbone joint reproduces exact pain")
        breath = st.checkbox("Sharp pain spikes when taking a deep breath or coughing")
        eat = st.checkbox("Pain worsens after eating or lying flat")
        burn = st.checkbox("Burning sensation rising in chest/throat")
        lift = st.checkbox("Recent heavy lifting, intense exercise, or chest trauma")
    with c2:
        audio_file = st.file_uploader("Upload Lung Auscultation Recording (.wav)", type=["wav"])

    if st.button("Run Cycle 2: Final Diagnosis"):
        df2 = pd.DataFrame([{
            'sternal_tenderness': int(tender), 'pain_on_deep_breath': int(breath),
            'worse_after_eating': int(eat), 'burning_sensation': int(burn),
            'recent_heavy_lifting': int(lift)
        }])
        tab_probs = model_stage2.predict_proba(df2)
        
        if audio_file is not None:
            spec = process_audio(audio_file)
            raw_prob = float(cnn_audio_model.predict(np.expand_dims(spec, axis=0), verbose=0))
            if raw_prob >= best_thresh:
                asthma_prob = 0.50 + 0.50 * ((raw_prob - best_thresh) / (1.0 - best_thresh + 1e-8))
            else:
                asthma_prob = 0.50 * (raw_prob / (best_thresh + 1e-8))
        else:
            spec = None
            asthma_prob = 0.05

        adj_tab = tab_probs * (1.0 - asthma_prob)
        all_probs = np.append(adj_tab, asthma_prob)
        all_probs = all_probs / np.sum(all_probs)
        
        classes = ['Costochondritis', 'GERD (Acid Reflux)', 'General Muscle Strain', 'Asthma / Wheeze']
        st.subheader("Diagnostic Probability Breakdown")
        for cls, p in zip(classes, all_probs):
            st.write(f"**{cls}:** {p*100:.1f}%")
            st.progress(float(p))

        # SHAP Explainability Chart
        shap_vals = explainer_stage2(df2)
        fig, ax = plt.subplots(figsize=(7, 3))
        vals = shap_vals[0, :, 0].values
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in vals]
        ax.barh(df2.columns, vals, color=colors)
        ax.set_title("SHAP Impact on Costochondritis Score")
        ax.axvline(0, color='black', linewidth=0.8)
        st.pyplot(fig)
