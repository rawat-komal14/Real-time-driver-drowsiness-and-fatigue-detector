import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def generate_synthetic_dataset(num_samples=1500):
    """
    Generates realistic synthetic facial landmark data based on spatial metrics:
    0: Alert (High EAR, Low MAR)
    1: Mild Fatigue (Medium EAR, Occasional Yawning)
    2: Critical Drowsiness (Low EAR / Closed Eyes, High MAR / Yawning)
    """
    np.random.seed(42)
    
    # Class 0: Alert (EAR ~ 0.28-0.38, MAR ~ 0.10-0.25)
    alert_ear = np.random.uniform(0.28, 0.38, num_samples // 3)
    alert_mar = np.random.uniform(0.10, 0.25, num_samples // 3)
    alert_labels = np.zeros(num_samples // 3, dtype=int)

    # Class 1: Mild Fatigue (EAR ~ 0.21-0.27, MAR ~ 0.20-0.45)
    mild_ear = np.random.uniform(0.21, 0.27, num_samples // 3)
    mild_mar = np.random.uniform(0.20, 0.45, num_samples // 3)
    mild_labels = np.ones(num_samples // 3, dtype=int)

    # Class 2: Critical Drowsiness (EAR ~ 0.08-0.20, MAR ~ 0.40-0.80)
    drowsy_ear = np.random.uniform(0.08, 0.20, num_samples // 3)
    drowsy_mar = np.random.uniform(0.40, 0.80, num_samples // 3)
    drowsy_labels = np.full(num_samples // 3, 2, dtype=int)

    # Combine data
    ear = np.concatenate([alert_ear, mild_ear, drowsy_ear])
    mar = np.concatenate([alert_mar, mild_mar, drowsy_mar])
    labels = np.concatenate([alert_labels, mild_labels, drowsy_labels])

    df = pd.DataFrame({'ear': ear, 'mar': mar, 'target': labels})
    return df

def train_and_save_model():
    print("1. Generating driver fatigue dataset...")
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

    print("2. Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    print("\n--- Model Evaluation Results ---")
    print(classification_report(y_test, y_pred, target_names=['Alert (0)', 'Mild (1)', 'Drowsy (2)']))

    # Save model and scaler files (.pkl)
    joblib.dump(model, 'drowsiness_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("3. Success! Saved 'drowsiness_model.pkl' and 'scaler.pkl'.")

if __name__ == "__main__":
    train_and_save_model()