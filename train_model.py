import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_dataset(num_samples=2100):
    """Generates a realistic driver fatigue dataset with natural feature overlap."""
    np.random.seed(42)
    n_per_class = num_samples // 3

    # Class 0: Alert
    alert_ear = np.random.normal(0.33, 0.045, n_per_class)
    alert_mar = np.random.normal(0.15, 0.045, n_per_class)
    alert_labels = np.zeros(n_per_class, dtype=int)

    # Class 1: Mild Fatigue
    mild_ear = np.random.normal(0.24, 0.045, n_per_class)
    mild_mar = np.random.normal(0.32, 0.065, n_per_class)
    mild_labels = np.ones(n_per_class, dtype=int)

    # Class 2: Critical Drowsiness
    drowsy_ear = np.random.normal(0.14, 0.045, n_per_class)
    drowsy_mar = np.random.normal(0.55, 0.080, n_per_class)
    drowsy_labels = np.full(n_per_class, 2, dtype=int)

    ear = np.concatenate([alert_ear, mild_ear, drowsy_ear])
    mar = np.concatenate([alert_mar, mild_mar, drowsy_mar])
    labels = np.concatenate([alert_labels, mild_labels, drowsy_labels])

    # 3% synthetic noise
    noise_indices = np.random.choice(
        len(labels), size=int(0.03 * len(labels)), replace=False
    )
    ear[noise_indices] += np.random.uniform(-0.06, 0.06, len(noise_indices))
    mar[noise_indices] += np.random.uniform(-0.08, 0.08, len(noise_indices))

    ear = np.clip(ear, 0.06, 0.42)
    mar = np.clip(mar, 0.06, 0.85)

    return pd.DataFrame({"ear": ear, "mar": mar, "target": labels})


def train_and_save_model():
    print("1. Generating synthetic driver fatigue dataset...")
    df = generate_synthetic_dataset()

    X = df[["ear", "mar"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("2. Training Random Forest & Logistic Regression Models...")

    # --- Train Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)

    # --- Train Logistic Regression ---
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)

    # Compute Metrics for Random Forest
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec, rf_rec, rf_f1, _ = precision_recall_fscore_support(
        y_test, rf_pred, average="weighted", zero_division=0
    )

    # Compute Metrics for Logistic Regression
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_prec, lr_rec, lr_f1, _ = precision_recall_fscore_support(
        y_test, lr_pred, average="weighted", zero_division=0
    )

    metrics_data = {
        "random_forest": {
            "accuracy": round(float(rf_acc * 100), 2),
            "precision": round(float(rf_prec * 100), 2),
            "recall": round(float(rf_rec * 100), 2),
            "f1_score": round(float(rf_f1 * 100), 2),
            "log_loss": round(float(log_loss(y_test, rf_proba)), 4),
        },
        "logistic_regression": {
            "accuracy": round(float(lr_acc * 100), 2),
            "precision": round(float(lr_prec * 100), 2),
            "recall": round(float(lr_rec * 100), 2),
            "f1_score": round(float(lr_f1 * 100), 2),
            "log_loss": round(float(log_loss(y_test, lr_proba)), 4),
        },
    }

    print("3. Generating Comparative Evaluation Plots...")

    # --- Generate Comparative Accuracy & Loss Bar Chart ---
    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor="#151515")
        ax1.set_facecolor("#151515")
        ax2.set_facecolor("#151515")

        models = ["Random Forest", "Logistic Reg."]
        accuracies = [metrics_data["random_forest"]["accuracy"], metrics_data["logistic_regression"]["accuracy"]]
        losses = [metrics_data["random_forest"]["log_loss"], metrics_data["logistic_regression"]["log_loss"]]

        # Accuracy Plot
        bars1 = ax1.bar(models, accuracies, color=["#EF4444", "#3B82F6"], width=0.5)
        ax1.set_ylabel("Test Accuracy (%)", color="#E5E7EB")
        ax1.set_title("Model Accuracy Comparison", fontweight="bold", color="#FFFFFF", pad=20)
        ax1.set_ylim(80, 103)
        ax1.grid(True, color="#444444", alpha=0.3, axis="y")
        for bar in bars1:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.8, f"{yval:.2f}%", ha="center", va="bottom", color="white", fontweight="bold")

        # Loss Plot (FIXED HEADROOM & TITLE PADDING)
        bars2 = ax2.bar(models, losses, color=["#EF4444", "#3B82F6"], width=0.5)
        ax2.set_ylabel("Log Loss", color="#E5E7EB")
        ax2.set_title("Model Log Loss Comparison", fontweight="bold", color="#FFFFFF", pad=20)
        ax2.set_ylim(0, max(losses) * 1.45)  # Added headroom so labels don't touch the title
        ax2.grid(True, color="#444444", alpha=0.3, axis="y")
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.003, f"{yval:.4f}", ha="center", va="bottom", color="white", fontweight="bold")

        fig.tight_layout()
        fig.savefig("model_comparison_graphs.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    # --- Generate Side-by-Side Confusion Matrices ---
    cm_rf = confusion_matrix(y_test, rf_pred, labels=[0, 1, 2])
    cm_lr = confusion_matrix(y_test, lr_pred, labels=[0, 1, 2])

    with plt.style.context("dark_background"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor="#151515")
        
        # RF Matrix
        ax1.set_facecolor("#151515")
        ax1.imshow(cm_rf, cmap="Reds")
        ax1.set_xticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])
        ax1.set_yticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])
        ax1.set_xlabel("Predicted Label", color="#E5E7EB")
        ax1.set_ylabel("True Label", color="#E5E7EB")
        ax1.set_title("Random Forest Confusion Matrix", color="#FFFFFF", fontweight="bold", pad=15)
        for i in range(3):
            for j in range(3):
                ax1.text(j, i, str(cm_rf[i, j]), ha="center", va="center", color="white", fontweight="bold")

        # LR Matrix
        ax2.set_facecolor("#151515")
        ax2.imshow(cm_lr, cmap="Blues")
        ax2.set_xticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])
        ax2.set_yticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])
        ax2.set_xlabel("Predicted Label", color="#E5E7EB")
        ax2.set_ylabel("True Label", color="#E5E7EB")
        ax2.set_title("Logistic Regression Confusion Matrix", color="#FFFFFF", fontweight="bold", pad=15)
        for i in range(3):
            for j in range(3):
                ax2.text(j, i, str(cm_lr[i, j]), ha="center", va="center", color="white", fontweight="bold")

        fig.tight_layout()
        fig.savefig("confusion_matrices_comparison.png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    print("\n--- Model Evaluation Results ---")
    print(f"Random Forest Accuracy       : {metrics_data['random_forest']['accuracy']}%")
    print(f"Logistic Regression Accuracy : {metrics_data['logistic_regression']['accuracy']}%")

    # Serialize Random Forest as primary deployment model alongside scaler
    joblib.dump(rf_model, "drowsiness_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print("Success! Saved primary model, scaler, metrics.json, and comparison charts.")


if __name__ == "__main__":
    train_and_save_model()