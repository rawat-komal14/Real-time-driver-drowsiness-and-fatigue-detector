from datetime import datetime
import json
import os
import cv2
import joblib
import numpy as np
import streamlit as st

try:
    import winsound
except ImportError:
    winsound = None

from feature_extractor import extract_features

st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="⚠️",
    layout="wide",
)

st.markdown(
    """
<style>
.stApp {
    background: #080808;
    color: #F9FAFB;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1550px;
}
section[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #292929;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #F3F4F6 !important;
}

h1, h2, h3, h4, h5, h6, 
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #FFFFFF !important;
}
p, span, label {
    color: #E4E4E7;
}

.header-box {
    background: linear-gradient(110deg, #171717, #101010);
    border: 1px solid #303030;
    border-left: 5px solid #DC2626;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 18px;
}
.header-title {
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: .4px;
}
.header-sub {
    font-size: 13px;
    color: #A1A1AA;
    margin-top: 5px;
}
.metric-tile {
    background: #151515;
    border: 1px solid #303030;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.metric-label {
    color: #A1A1AA;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.metric-value {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: 800;
    margin-top: 4px;
}
.status-card {
    width: 100%;
    padding: 14px 10px;
    border-radius: 9px;
    font-size: 15px;
    font-weight: 850;
    text-align: center;
    letter-spacing: .7px;
    margin-bottom: 12px;
    border: 1px solid #333333;
}
.status-alert {
    background: #172019;
    color: #86EFAC;
    border-color: #166534;
}
.status-mild {
    background: #29230D;
    color: #FACC15;
    border-color: #A16207;
}
.status-critical {
    background: #3A1115;
    color: #FCA5A5;
    border-color: #DC2626;
}
.status-off {
    background: #18181B;
    color: #A1A1AA;
    border-color: #3F3F46;
}
.alarm-box {
    background: #991B1B;
    color: #FFFFFF;
    padding: 14px 10px;
    border-radius: 9px;
    text-align: center;
    font-weight: 900;
    font-size: 13px;
    margin-top: 10px;
    border: 1px solid #EF4444;
}
hr {
    border-color: #303030;
}

[data-baseweb="tab"] p, [data-baseweb="tab"] span {
    color: #D4D4D8 !important;
    font-weight: 600 !important;
}
[data-baseweb="tab"][aria-selected="true"] span, [data-baseweb="tab"][aria-selected="true"] p {
    color: #EF4444 !important;
    font-weight: 700 !important;
}

div[data-testid="stMetricValue"] > div {
    font-size: 1.22rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    color: #FFFFFF !important;
}
div[data-testid="stMetricLabel"] > div {
    font-size: 0.78rem !important;
    color: #D4D4D8 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Header Banner
st.markdown(
    """
<div class="header-box">
    <div class="header-title">🚗 DRIVER DROWSINESS DETECTION SYSTEM</div>
    <div class="header-sub">
        Real-time facial landmark analysis · Multi-model safety monitoring & performance analytics
    </div>
</div>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_ml_assets():
    model = joblib.load("drowsiness_model.pkl")
    scaler = joblib.load("scaler.pkl")

    try:
        with open("metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None

    return model, scaler, metrics

try:
    model, scaler, metrics = load_ml_assets()
except Exception as e:
    st.error("Model files could not be loaded. Run 'python train_model.py' first.")
    st.exception(e)
    st.stop()

# Sidebar Controls & Settings
st.sidebar.markdown("## ⚙️ Control Panel")
run_app = st.sidebar.toggle("Start Camera", key="run_app")
st.sidebar.caption("Toggle to activate live webcam inference.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Calibration & Settings")

ear_threshold = st.sidebar.slider(
    "EAR Closure Threshold", 
    min_value=0.12, max_value=0.25, value=0.18, step=0.01,
    help="Adjust sensitivity for eye closure detection."
)
alarm_delay_frames = st.sidebar.slider(
    "Alarm Delay (Frames)", 
    min_value=1, max_value=15, value=5, step=1,
    help="Number of consecutive frames required to trigger alert."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📌 Reference Metrics")
st.sidebar.markdown(
    "- **EAR (Eye Aspect Ratio):** Drops below threshold during eye closure.\n"
    "- **MAR (Mouth Aspect Ratio):** Increases during yawning and fatigue."
)

# Main Application Tabs for Structured UX
tab_live, tab_metrics, tab_matrix = st.tabs([
    "🚗 Live Monitoring", 
    "📊 Model Performance & Charts", 
    "🧩 Confusion Matrices"
])

# TAB 1: LIVE MONITORING
with tab_live:
    st.markdown("### Real-Time Webcam Inference & Active Safety Monitoring")
    st.markdown("Stream active video analysis, eye/mouth tracking parameters, and instant hazard alerts.")
    
    video_col, monitor_col = st.columns([1.7, 1], gap="large")

    with video_col:
        with st.container(border=True):
            st.markdown("#### 🎥 Live Camera Stream")
            video_placeholder = st.empty()
            if not run_app:
                video_placeholder.info("Camera is on standby. Switch on 'Start Camera' in the sidebar.")

    with monitor_col:
        with st.container(border=True):
            st.markdown("#### 🛡️ Driver Status Panel")

            status_placeholder = st.empty()
            status_placeholder.markdown('<div class="status-card status-off">SYSTEM STANDBY</div>', unsafe_allow_html=True)

            ear_card = st.empty()
            ear_bar = st.empty()
            mar_card = st.empty()
            mar_bar = st.empty()
            alarm_placeholder = st.empty()

            if not run_app:
                ear_card.markdown(
                    '<div class="metric-tile"><div class="metric-label">Eye Aspect Ratio (EAR)</div><div class="metric-value">--</div></div>',
                    unsafe_allow_html=True
                )
                mar_card.markdown(
                    '<div class="metric-tile"><div class="metric-label">Mouth Aspect Ratio (MAR)</div><div class="metric-value">--</div></div>',
                    unsafe_allow_html=True
                )

# TAB 2: MODEL PERFORMANCE
with tab_metrics:
    st.markdown("### Model Performance Evaluation")
    st.markdown("Comparative accuracy, precision, recall, and log loss metrics across algorithms.")

    if metrics and "random_forest" in metrics:
        rf = metrics["random_forest"]
        lr = metrics["logistic_regression"]
        svm = metrics.get("svm", {"accuracy": "--", "precision": "--", "recall": "--", "f1_score": "--"})

        col_rf, col_lr, col_svm = st.columns(3, gap="medium")

        with col_rf:
            with st.container(border=True):
                st.markdown("#### 🌲 Random Forest")
                r1_1, r1_2 = st.columns(2)
                r1_1.metric("Accuracy", f"{rf['accuracy']}%")
                r1_2.metric("Precision", f"{rf['precision']}%")
                
                r2_1, r2_2 = st.columns(2)
                r2_1.metric("Recall", f"{rf['recall']}%")
                r2_2.metric("F1 Score", f"{rf['f1_score']}%")

        with col_lr:
            with st.container(border=True):
                st.markdown("#### 📈 Logistic Regression")
                r1_1, r1_2 = st.columns(2)
                r1_1.metric("Accuracy", f"{lr['accuracy']}%")
                r1_2.metric("Precision", f"{lr['precision']}%")
                
                r2_1, r2_2 = st.columns(2)
                r2_1.metric("Recall", f"{lr['recall']}%")
                r2_2.metric("F1 Score", f"{lr['f1_score']}%")

        with col_svm:
            with st.container(border=True):
                st.markdown("#### 🎯 Support Vector Machine")
                r1_1, r1_2 = st.columns(2)
                r1_1.metric("Accuracy", f"{svm['accuracy']}%")
                r1_2.metric("Precision", f"{svm['precision']}%")
                
                r2_1, r2_2 = st.columns(2)
                r2_1.metric("Recall", f"{svm['recall']}%")
                r2_2.metric("F1 Score", f"{svm['f1_score']}%")
    else:
        st.warning("Metrics file not found. Run 'python train_model.py'.")

    st.markdown("---")
    st.markdown("### Comparative Performance Charts")
    if os.path.exists("model_comparison_graphs.png"):
        st.image("model_comparison_graphs.png")
    else:
        st.info("Run train_model.py to generate comparative graphs.")

# TAB 3: CONFUSION MATRICES
with tab_matrix:
    st.markdown("### Classification Breakdown (Confusion Matrices)")
    st.markdown("Detailed true label vs. predicted label comparisons across all three trained algorithms.")

    if os.path.exists("confusion_matrices_comparison.png"):
        st.image("confusion_matrices_comparison.png")
    else:
        st.info("Run train_model.py to generate confusion matrices.")

# LIVE WEBCAM EXECUTION LOOP (High-Performance Blocking While Loop)
if run_app:
    cap = cv2.VideoCapture(0)

    closed_eye_counter = 0
    drowsy_frame_counter = 0
    alarm_active = False

    if not cap.isOpened():
        st.error("Unable to access the camera. Check camera permissions and close other apps using it.")
    else:
        try:
            while run_app:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera stream unavailable.")
                    break

                # Downscale to 640x480 for lightning-fast face landmark processing
                frame = cv2.resize(frame, (640, 480))

                frame, ear, mar, face_found = extract_features(frame)

                if face_found:
                    features = np.array([[ear, mar]])
                    scaled_features = scaler.transform(features)
                    pred = int(model.predict(scaled_features)[0])

                    ear_card.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="metric-label">Eye Aspect Ratio (EAR)</div>'
                        f'<div class="metric-value">{ear:.3f}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    ear_bar.progress(min(max(float(ear / 0.4), 0.0), 1.0))

                    mar_card.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="metric-label">Mouth Aspect Ratio (MAR)</div>'
                        f'<div class="metric-value">{mar:.3f}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    mar_bar.progress(min(max(float(mar / 0.8), 0.0), 1.0))

                    if ear < ear_threshold:
                        closed_eye_counter += 1
                    else:
                        closed_eye_counter = 0

                    if pred == 2:
                        drowsy_frame_counter += 1
                    else:
                        drowsy_frame_counter = 0

                    sustained_eye_closure = (closed_eye_counter >= alarm_delay_frames)
                    sustained_drowsiness = (drowsy_frame_counter >= alarm_delay_frames)

                    if sustained_eye_closure or sustained_drowsiness:
                        status_txt = "CRITICAL DROWSINESS"
                        status_color = (0, 0, 255)
                        status_class = "status-critical"

                        alarm_placeholder.markdown(
                            '<div class="alarm-box">⚠️ CRITICAL DROWSINESS ALARM ⚠️</div>',
                            unsafe_allow_html=True,
                        )

                        if not alarm_active:
                            if winsound:
                                winsound.Beep(2500, 700)
                            
                            os.makedirs("captures", exist_ok=True)
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            snapshot_path = os.path.join("captures", f"drowsy_alert_{timestamp}.jpg")
                            cv2.imwrite(snapshot_path, frame)
                            st.toast(f"📸 Evidence snapshot saved: {snapshot_path}", icon="🚨")

                        alarm_active = True

                    elif pred == 1:
                        status_txt = "MILD FATIGUE"
                        status_color = (0, 220, 255)
                        status_class = "status-mild"
                        alarm_placeholder.empty()
                        alarm_active = False

                    else:
                        status_txt = "ALERT"
                        status_color = (0, 220, 0)
                        status_class = "status-alert"
                        alarm_placeholder.empty()
                        alarm_active = False

                    status_placeholder.markdown(
                        f'<div class="status-card {status_class}">STATUS: {status_txt}</div>',
                        unsafe_allow_html=True,
                    )

                    cv2.putText(frame, f"STATE: {status_txt}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    cv2.putText(frame, f"EAR: {ear:.3f}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                else:
                    closed_eye_counter = 0
                    drowsy_frame_counter = 0
                    alarm_active = False

                    status_placeholder.markdown(
                        '<div class="status-card status-off">SEARCHING FOR FACE...</div>',
                        unsafe_allow_html=True,
                    )

                    alarm_placeholder.empty()
                    ear_card.markdown(
                        '<div class="metric-tile"><div class="metric-label">Eye Aspect Ratio (EAR)</div><div class="metric-value">Waiting...</div></div>',
                        unsafe_allow_html=True
                    )
                    mar_card.markdown(
                        '<div class="metric-tile"><div class="metric-label">Mouth Aspect Ratio (MAR)</div><div class="metric-value">Waiting...</div></div>',
                        unsafe_allow_html=True
                    )
                    ear_bar.empty()
                    mar_bar.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")

        finally:
            cap.release()