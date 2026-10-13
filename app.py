import json
import cv2
import joblib

# Optional import for audio alarms on Windows
try:
    import winsound
except ImportError:
    winsound = None

import numpy as np
import streamlit as st
from feature_extractor import extract_features

# Page Setup
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="⚠️",
    layout="wide",
)

# High-Contrast Black, White & Red Styling
st.markdown(
    """
<style>
    /* Main Background */
    .stApp {
        background-color: #050505;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Top Header */
    .header-box {
        background-color: #121212;
        border: 1px solid #262626;
        border-left: 5px solid #dc2626;
        padding: 18px 24px;
        border-radius: 4px;
        margin-bottom: 24px;
    }
    
    .header-title {
        font-size: 22px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    
    .header-sub {
        font-size: 13px;
        color: #a1a1aa;
        margin-top: 4px;
    }

    /* Metric Display Cards */
    .metric-card {
        background-color: #121212;
        border: 1px solid #262626;
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .metric-title {
        font-size: 12px;
        font-weight: 700;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-num {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin-top: 6px;
    }

    /* Status Indicators */
    .status-card {
        width: 100%;
        padding: 14px;
        border-radius: 4px;
        font-size: 16px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .status-alert {
        background-color: #18181b;
        color: #ffffff;
        border: 2px solid #27272a;
    }

    .status-mild {
        background-color: #27272a;
        color: #facc15;
        border: 2px solid #eab308;
    }

    .status-critical {
        background-color: #7f1d1d;
        color: #ffffff;
        border: 2px solid #dc2626;
    }

    .status-off {
        background-color: #121212;
        color: #52525b;
        border: 1px solid #262626;
    }

    /* Alarm Banner */
    .alarm-box {
        background-color: #dc2626;
        color: #ffffff;
        padding: 16px;
        border-radius: 4px;
        text-align: center;
        font-weight: 900;
        font-size: 16px;
        letter-spacing: 1.5px;
        margin-top: 12px;
        border: 1px solid #f87171;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Top Bar Header
st.markdown(
    """
<div class="header-box">
    <div class="header-title">DRIVER DROWSINESS DETECTION SYSTEM</div>
    <div class="header-sub">Real-Time Facial Landmark Analysis & Safety Monitoring</div>
</div>
""",
    unsafe_allow_html=True,
)


# Load Model Assets & Metrics
@st.cache_resource
def load_ml_assets():
    model = joblib.load('drowsiness_model.pkl')
    scaler = joblib.load('scaler.pkl')

    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None

    return model, scaler, metrics


try:
    model, scaler, metrics = load_ml_assets()
except Exception:
    st.error("Model files not found. Run 'python train_model.py' first.")
    st.stop()

# Sidebar Setup
st.sidebar.markdown('### Controls')
run_app = st.sidebar.toggle('Start Camera', value=False)

st.sidebar.markdown('---')

# Display Model Training Performance Metrics & Confusion Matrix in Sidebar
st.sidebar.markdown('### 📊 Model Performance')
if metrics:
    m_col1, m_col2 = st.sidebar.columns(2)
    m_col1.metric('Accuracy', f"{metrics['accuracy']}%")
    m_col2.metric('Precision', f"{metrics['precision']}%")

    m_col3, m_col4 = st.sidebar.columns(2)
    m_col3.metric('Recall', f"{metrics['recall']}%")
    m_col4.metric('F1-Score', f"{metrics['f1_score']}%")
    
    st.sidebar.markdown('#### Confusion Matrix')
    try:
        st.sidebar.image('confusion_matrix.png')
    except FileNotFoundError:
        st.sidebar.info("Run 'python train_model.py' to generate confusion matrix.")
else:
    st.sidebar.info("Re-run 'python train_model.py' to generate metrics.")

st.sidebar.markdown('---')
st.sidebar.markdown('### Reference Metrics')
st.sidebar.markdown("""
- **EAR (Eye Closure):** Drops below 0.20 when closing eyes.
- **MAR (Yawning):** Exceeds 0.40 when yawning.
""")

# Dashboard Grid
col_video, col_data = st.columns([2.3, 1.1])

with col_video:
    st.markdown('##### Live Camera Feed')
    video_placeholder = st.empty()

with col_data:
    st.markdown('##### System Metrics')
    status_placeholder = st.empty()
    status_placeholder.markdown(
        '<div class="status-card status-off">SYSTEM STANDBY</div>',
        unsafe_allow_html=True,
    )

    ear_card = st.empty()
    ear_bar = st.empty()

    mar_card = st.empty()
    mar_bar = st.empty()

    alarm_placeholder = st.empty()

# Frame Buffer for Continuous Drowsiness Verification
drowsy_frame_counter = 0
ALARM_THRESHOLD_FRAMES = 8

if run_app:
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret:
            st.error('Camera stream unavailable.')
            break

        # Step 1: Feature Extraction
        frame, ear, mar, face_found = extract_features(frame)

        if face_found:
            # Step 2: Model Inference
            features = np.array([[ear, mar]])
            scaled_features = scaler.transform(features)
            pred = model.predict(scaled_features)[0]

            # Render Cards
            ear_card.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-title">Eye Aspect Ratio (EAR)</div>
                <div class="metric-num">{ear:.3f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            ear_bar.progress(min(max(float(ear / 0.4), 0.0), 1.0))

            mar_card.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-title">Mouth Aspect Ratio (MAR)</div>
                <div class="metric-num">{mar:.3f}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            mar_bar.progress(min(max(float(mar / 0.8), 0.0), 1.0))

            # Driver Status Logic
            if pred == 0:
                drowsy_frame_counter = 0
                status_placeholder.markdown(
                    '<div class="status-card status-alert">STATUS: ALERT</div>',
                    unsafe_allow_html=True,
                )
                alarm_placeholder.empty()
                status_txt, status_color = 'ALERT', (0, 255, 0)

            elif pred == 1:
                drowsy_frame_counter = 0
                status_placeholder.markdown(
                    '<div class="status-card status-mild">STATUS: MILD FATIGUE</div>',
                    unsafe_allow_html=True,
                )
                alarm_placeholder.empty()
                status_txt, status_color = 'MILD FATIGUE', (0, 255, 255)

            elif pred == 2:
                drowsy_frame_counter += 1
                status_placeholder.markdown(
                    '<div class="status-card status-critical">STATUS: DROWSY</div>',
                    unsafe_allow_html=True,
                )
                status_txt, status_color = 'CRITICAL DROWSINESS', (0, 0, 255)

                # TRIGGER HARDWARE AUDIO ALARM + VISUAL WARNING
                if drowsy_frame_counter >= ALARM_THRESHOLD_FRAMES:
                    alarm_placeholder.markdown(
                        """
                    <div class="alarm-box">
                        ⚠️ CRITICAL DROWSINESS ALARM ⚠️
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    if winsound:
                        winsound.Beep(2500, 300)

            # Draw status on OpenCV video frame
            cv2.putText(
                frame,
                f'STATE: {status_txt}',
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2,
            )

        else:
            status_placeholder.markdown(
                '<div class="status-card status-off">SEARCHING FOR FACE...</div>',
                unsafe_allow_html=True,
            )
            alarm_placeholder.empty()

        # Display Frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

    cap.release()