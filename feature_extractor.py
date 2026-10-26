import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp


try:
    # Legacy Solutions API (MediaPipe < 0.10.14)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    USE_TASKS_API = False
except AttributeError:
    # Modern Tasks API (MediaPipe 0.10.14+ / Python 3.13)
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MODEL_PATH = "face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    USE_TASKS_API = True

# Landmark indices for Left Eye, Right Eye, and Mouth
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 0, 17, 84, 314]


def euclidean_distance(point1, point2):
    """Calculates 2D Euclidean distance between two landmark coordinate points."""
    return np.linalg.norm(point1 - point2)


def calculate_ear(eye_landmarks):
    """Calculates Eye Aspect Ratio (EAR). Drops when eyes close."""
    v1 = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    h = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (v1 + v2) / (2.0 * h)


def calculate_mar(mouth_landmarks):
    """Calculates Mouth Aspect Ratio (MAR). Increases during yawning."""
    v = euclidean_distance(mouth_landmarks[2], mouth_landmarks[3])
    h = euclidean_distance(mouth_landmarks[0], mouth_landmarks[1])
    return v / h


def extract_features(frame):
    """Processes a video frame and returns extracted facial features:

    (processed_frame, average_ear, mar, face_detected_bool)
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not USE_TASKS_API:
        results = face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return frame, 0.0, 0.0, False
        face_landmarks = results.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)
        if not detection_result.face_landmarks:
            return frame, 0.0, 0.0, False
        face_landmarks = detection_result.face_landmarks[0]

    def get_coords(indices):
        return np.array([
            (int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h))
            for idx in indices
        ])

    left_eye = get_coords(LEFT_EYE_INDICES)
    right_eye = get_coords(RIGHT_EYE_INDICES)
    mouth = get_coords(MOUTH_INDICES)

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    mar = calculate_mar(mouth)

    return frame, avg_ear, mar, True


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Starting webcam test... Press 'q' on your keyboard to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, ear, mar, detected = extract_features(frame)

        if detected:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Feature Extractor Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()