import json
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
    padding-top: 1.6rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}
section[data-testid="stSidebar"] {
    background: #111114;
    border-right: 1px solid #292929;
}
.header-box {
    background: linear-gradient(110deg, #171717, #101010);
    border: 1px solid #303030;
    border-left: 5px solid #DC2626;
    padding: 22px 26px;
    border-radius: 12px;
    margin-bottom: 22px;
}
.header-title {
    font-size: 27px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: .4px;
}
.header-sub {
    font-size: 14px;
    color: #A1A1AA;
    margin-top: 7px;
}
.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #F9FAFB;
    margin: 10px 0 4px 0;
}
.section-subtitle {
    color: #A1A1AA;
    font-size: 13px;
    margin-bottom: 14px;
}
.metric-tile {
    background: #151515;
    border: 1px solid #303030;
    border-radius: 10px;
    padding: 17px 18px;
    min-height: 105px;
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
    font-size: 29px;
    font-weight: 800;
    margin-top: 8px;
}
.panel {
    background: #121212;
    border: 1px solid #303030;
    border-radius: 12px;
    padding: 15px;
}
.status-card {
    width: 100%;
    padding: 15px 10px;
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
    padding: 15px 10px;
    border-radius: 9px;
    text-align: center;
    font-weight: 900;
    font-size: 14px;
    margin-top: 12px;
    border: 1px solid #EF4444;
}
div[data-testid="stMetric"] {
    background: #151515;
    border: 1px solid #303030;
    padding: 12px 14px;
    border-radius: 10px;
}
div[data-testid="stMetricLabel"] {
    color: #A1A1AA;
}
hr {
    border-color: #303030;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header-box">
    <div class="header-title">DRIVER DROWSINESS DETECTION SYSTEM</div>
    <div class="header-sub">
        Real-time facial landmark analysis · Driver safety monitoring
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
    st.error(
        "Model files could not be loaded. "
        "Run 'python train_model.py' first."
    )
    st.exception(e)
    st.stop()


st.sidebar.markdown("## ⚙️ Controls")
run_app = st.sidebar.toggle("Start Camera", value=False)
st.sidebar.caption("Enable the camera to begin live monitoring.")

st.sidebar.markdown("---")
st.sidebar.markdown("## Reference Metrics")
st.sidebar.markdown(
    "- **EAR:** Eye Aspect Ratio; decreases as eyes close.\n"
    "- **MAR:** Mouth Aspect Ratio; increases during yawning."
)

st.sidebar.markdown("---")
st.sidebar.markdown("## Class Labels")
st.sidebar.markdown(
    "- 🟢 **Alert** — alert prediction\n"
    "- 🟡 **Mild fatigue** — possible fatigue\n"
    "- 🔴 **Drowsy** — drowsiness prediction"
)

st.sidebar.caption(
    "Experimental student project. Do not rely on it as the sole "
    "driving-safety mechanism."
)


st.markdown(
    '<div class="section-heading">Model Performance Overview</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Evaluation results from the synthetic test dataset.</div>',
    unsafe_allow_html=True
)

if metrics:
    metric_cols = st.columns(4)

    metric_items = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1 Score", metrics["f1_score"]),
    ]

    for col, (label, value) in zip(metric_cols, metric_items):
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
else:
    st.warning(
        "Metrics file not found. Run 'python train_model.py'."
    )


st.markdown("")
st.markdown(
    '<div class="section-heading">Model Training Analysis</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Test accuracy and log loss as the number of Random Forest trees increases.'
    '</div>',
    unsafe_allow_html=True
)

graph_col1, graph_col2 = st.columns(2, gap="large")

with graph_col1:
    with st.container(border=True):
        st.markdown("#### Accuracy")
        try:
            st.image("accuracy_graph.png", use_container_width=True)
        except Exception:
            st.info("Run train_model.py to generate the accuracy graph.")

with graph_col2:
    with st.container(border=True):
        st.markdown("#### Log Loss")
        try:
            st.image("loss_graph.png", use_container_width=True)
        except Exception:
            st.info("Run train_model.py to generate the loss graph.")


st.markdown(
    '<div class="section-heading">Confusion Matrix</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Correct and incorrect predictions for the three classes.</div>',
    unsafe_allow_html=True
)

cm_col1, cm_col2 = st.columns([1, 1.7], gap="large")

with cm_col1:
    try:
        st.image("confusion_matrix.png", use_container_width=True)
    except Exception:
        st.info("Run train_model.py to generate the confusion matrix.")

with cm_col2:
    st.markdown(
        """
        <div class="panel">
            <b>How to read this matrix</b>
            <p style="color:#A1A1AA; font-size:14px;">
            Rows show the true class, and columns show the predicted class.
            Values on the diagonal are correct classifications. Values outside
            the diagonal are misclassifications.
            </p>
            <hr>
            <span style="color:#86EFAC;">Alert</span> ·
            <span style="color:#FACC15;">Mild fatigue</span> ·
            <span style="color:#FCA5A5;">Drowsy</span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")
st.markdown(
    '<div class="section-heading">Live Driver Monitoring</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">'
    'Monitor eye and mouth measurements and the current driver status.'
    '</div>',
    unsafe_allow_html=True
)

video_col, monitor_col = st.columns([1.8, 1], gap="large")

with video_col:
    with st.container(border=True):
        st.markdown("#### Camera Feed")
        video_placeholder = st.empty()
        video_placeholder.info(
            "Camera is on standby. Switch on Start Camera in the sidebar."
        )

with monitor_col:
    with st.container(border=True):
        st.markdown("#### Driver Status")

        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<div class="status-card status-off">SYSTEM STANDBY</div>',
            unsafe_allow_html=True
        )

        ear_card = st.empty()
        ear_bar = st.empty()
        mar_card = st.empty()
        mar_bar = st.empty()
        alarm_placeholder = st.empty()


if run_app:
    cap = cv2.VideoCapture(0)

    EYE_CLOSED_THRESHOLD = 0.20
    ALARM_THRESHOLD_FRAMES = 8

    closed_eye_counter = 0
    drowsy_frame_counter = 0
    alarm_active = False

    if not cap.isOpened():
        st.error(
            "Unable to access the camera. Check camera permissions "
            "and close other applications using it."
        )
    else:
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    st.error("Camera stream unavailable.")
                    break

                frame, ear, mar, face_found = extract_features(frame)

                if face_found:
                    features = np.array([[ear, mar]])
                    scaled_features = scaler.transform(features)
                    pred = int(model.predict(scaled_features)[0])

                    print(
                        f"EAR: {ear:.3f} | MAR: {mar:.3f} | "
                        f"Prediction: {pred}"
                    )

                    ear_card.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="metric-label">Eye Aspect Ratio (EAR)</div>'
                        f'<div class="metric-value">{ear:.3f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    ear_bar.progress(
                        min(max(float(ear / 0.4), 0.0), 1.0)
                    )

                    mar_card.markdown(
                        f'<div class="metric-tile">'
                        f'<div class="metric-label">Mouth Aspect Ratio (MAR)</div>'
                        f'<div class="metric-value">{mar:.3f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    mar_bar.progress(
                        min(max(float(mar / 0.8), 0.0), 1.0)
                    )

                    if ear < EYE_CLOSED_THRESHOLD:
                        closed_eye_counter += 1
                    else:
                        closed_eye_counter = 0

                    if pred == 2:
                        drowsy_frame_counter += 1
                    else:
                        drowsy_frame_counter = 0

                    sustained_eye_closure = (
                        closed_eye_counter >= ALARM_THRESHOLD_FRAMES
                    )
                    sustained_drowsiness = (
                        drowsy_frame_counter >= ALARM_THRESHOLD_FRAMES
                    )

                    if sustained_eye_closure or sustained_drowsiness:
                        status_txt = "CRITICAL DROWSINESS"
                        status_color = (0, 0, 255)
                        status_class = "status-critical"

                        alarm_placeholder.markdown(
                            '<div class="alarm-box">'
                            '⚠️ CRITICAL DROWSINESS ALARM ⚠️'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        if not alarm_active and winsound:
                            winsound.Beep(2500, 700)

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
                        f'<div class="status-card {status_class}">'
                        f'STATUS: {status_txt}</div>',
                        unsafe_allow_html=True
                    )

                    cv2.putText(
                        frame,
                        f"STATE: {status_txt}",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        status_color,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"EAR: {ear:.3f}",
                        (25, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Closed frames: {closed_eye_counter}",
                        (25, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                else:
                    closed_eye_counter = 0
                    drowsy_frame_counter = 0
                    alarm_active = False

                    status_placeholder.markdown(
                        '<div class="status-card status-off">'
                        'SEARCHING FOR FACE...</div>',
                        unsafe_allow_html=True
                    )

                    alarm_placeholder.empty()
                    ear_card.info("EAR: waiting for face")
                    mar_card.info("MAR: waiting for face")
                    ear_bar.empty()
                    mar_bar.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                video_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_container_width=True
                )

        finally:
            cap.release()

else:
    st.caption(
        "Camera is off. Model metrics, graphs, and confusion matrix "
        "remain available above."
    )
