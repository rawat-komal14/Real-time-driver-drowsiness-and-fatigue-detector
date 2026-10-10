import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_dataset(num_samples=2100):
    """Generates realistic synthetic facial landmark data with natural feature overlap,

    facial variance, and webcam sensor noise to prevent overfitting.
    """
    np.random.seed(42)
    n_per_class = num_samples // 3

    # Class 0: Alert (Mean EAR: 0.32, Mean MAR: 0.18)
    alert_ear = np.random.normal(loc=0.32, scale=0.045, size=n_per_class)
    alert_mar = np.random.normal(loc=0.18, scale=0.055, size=n_per_class)
    alert_labels = np.zeros(n_per_class, dtype=int)

    # Class 1: Mild Fatigue (Mean EAR: 0.24, Mean MAR: 0.33) - Significant overlap zone
    mild_ear = np.random.normal(loc=0.24, scale=0.050, size=n_per_class)
    mild_mar = np.random.normal(loc=0.33, scale=0.075, size=n_per_class)
    mild_labels = np.ones(n_per_class, dtype=int)

    # Class 2: Critical Drowsiness (Mean EAR: 0.15, Mean MAR: 0.55)
    drowsy_ear = np.random.normal(loc=0.15, scale=0.055, size=n_per_class)
    drowsy_mar = np.random.normal(loc=0.55, scale=0.095, size=n_per_class)
    drowsy_labels = np.full(n_per_class, 2, dtype=int)

    # Combine classes
    ear = np.concatenate([alert_ear, mild_ear, drowsy_ear])
    mar = np.concatenate([alert_mar, mild_mar, drowsy_mar])
    labels = np.concatenate([alert_labels, mild_labels, drowsy_labels])

    # Introduce 3% synthetic noise/outliers (e.g., sudden talking, camera jitter, or rapid blinks)
    noise_indices = np.random.choice(
        num_samples, size=int(0.03 * num_samples), replace=False
    )
    ear[noise_indices] += np.random.uniform(-0.08, 0.08, size=len(noise_indices))
    mar[noise_indices] += np.random.uniform(-0.10, 0.10, size=len(noise_indices))

    # Clip values to realistic biometric thresholds
    ear = np.clip(ear, 0.05, 0.45)
    mar = np.clip(mar, 0.05, 0.90)

    df = pd.DataFrame({'ear': ear, 'mar': mar, 'target': labels})
    return df


def train_and_save_model():
    print("1. Generating realistic driver fatigue dataset with noise...")
    df = generate_synthetic_dataset()

    X = df[['ear', 'mar']]
    y = df['target']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("2. Training Regularized Random Forest Classifier...")
    # Regularized hyperparameters (max_depth, min_samples_leaf) prevent overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    metrics_data = {
        'accuracy': round(float(acc * 100), 2),
        'precision': round(float(prec * 100), 2),
        'recall': round(float(rec * 100), 2),
        'f1_score': round(float(f1 * 100), 2),
    }

    print('\n--- Model Evaluation Results ---')
    print(f"Accuracy : {metrics_data['accuracy']}%")
    print(f"Precision: {metrics_data['precision']}%")
    print(f"Recall   : {metrics_data['recall']}%")
    print(f"F1-Score : {metrics_data['f1_score']}%")

    # Save model, scaler, and metrics
    joblib.dump(model, 'drowsiness_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f)

    print(
        "3. Success! Saved 'drowsiness_model.pkl', 'scaler.pkl', and 'metrics.json'."
    )


if __name__ == '__main__':
    train_and_save_model()