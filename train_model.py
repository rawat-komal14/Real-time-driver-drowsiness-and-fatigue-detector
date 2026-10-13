import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_synthetic_dataset(num_samples=2100):
    np.random.seed(42)
    n_per_class = num_samples // 3

    alert_ear = np.random.normal(0.32, 0.045, n_per_class)
    alert_mar = np.random.normal(0.18, 0.055, n_per_class)
    alert_labels = np.zeros(n_per_class, dtype=int)

    mild_ear = np.random.normal(0.24, 0.050, n_per_class)
    mild_mar = np.random.normal(0.33, 0.075, n_per_class)
    mild_labels = np.ones(n_per_class, dtype=int)

    drowsy_ear = np.random.normal(0.15, 0.055, n_per_class)
    drowsy_mar = np.random.normal(0.55, 0.095, n_per_class)
    drowsy_labels = np.full(n_per_class, 2, dtype=int)

    ear = np.concatenate([alert_ear, mild_ear, drowsy_ear])
    mar = np.concatenate([alert_mar, mild_mar, drowsy_mar])
    labels = np.concatenate([alert_labels, mild_labels, drowsy_labels])

    noise_indices = np.random.choice(
        len(labels), size=int(0.03 * len(labels)), replace=False
    )
    ear[noise_indices] += np.random.uniform(-0.08, 0.08, len(noise_indices))
    mar[noise_indices] += np.random.uniform(-0.10, 0.10, len(noise_indices))

    ear = np.clip(ear, 0.05, 0.45)
    mar = np.clip(mar, 0.05, 0.90)

    return pd.DataFrame({"ear": ear, "mar": mar, "target": labels})


def style_axes(ax):
    ax.set_facecolor("#151515")
    ax.tick_params(colors="#E5E7EB")
    ax.xaxis.label.set_color("#E5E7EB")
    ax.yaxis.label.set_color("#E5E7EB")
    ax.title.set_color("#FFFFFF")

    for spine in ax.spines.values():
        spine.set_color("#555555")

    ax.grid(True, color="#444444", alpha=0.5)


def save_dark_graph(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#151515")
    style_axes(ax)

    ax.plot(
        x, y,
        color="#EF4444",
        marker="o",
        markersize=4,
        linewidth=2
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(
        filename,
        dpi=160,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )
    plt.close(fig)


def train_and_save_model():
    print("1. Generating synthetic driver fatigue dataset...")
    df = generate_synthetic_dataset()

    X = df[["ear", "mar"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("2. Generating accuracy and loss graph data...")

    tree_counts = list(range(1, 101, 5))
    accuracy_values = []
    loss_values = []

    for n_trees in tree_counts:
        temp_model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        temp_model.fit(X_train_scaled, y_train)

        predictions = temp_model.predict(X_test_scaled)
        probabilities = temp_model.predict_proba(X_test_scaled)

        accuracy_values.append(
            accuracy_score(y_test, predictions) * 100
        )

        loss_values.append(
            log_loss(y_test, probabilities, labels=[0, 1, 2])
        )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics_data = {
        "accuracy": round(float(acc * 100), 2),
        "precision": round(float(prec * 100), 2),
        "recall": round(float(rec * 100), 2),
        "f1_score": round(float(f1 * 100), 2)
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#151515")
    ax.set_facecolor("#151515")
    ax.imshow(cm, cmap="Reds")

    ax.set_xticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])
    ax.set_yticks([0, 1, 2], labels=["Alert", "Mild", "Drowsy"])

    ax.set_xlabel("Predicted Label", color="#E5E7EB")
    ax.set_ylabel("True Label", color="#E5E7EB")
    ax.set_title(
        "Confusion Matrix",
        color="#FFFFFF",
        pad=12,
        fontweight="bold"
    )
    ax.tick_params(colors="#E5E7EB")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )

    fig.tight_layout()
    fig.savefig(
        "confusion_matrix.png",
        dpi=160,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight"
    )
    plt.close(fig)

    save_dark_graph(
        tree_counts,
        accuracy_values,
        "Number of Trees",
        "Test Accuracy (%)",
        "Random Forest Accuracy",
        "accuracy_graph.png"
    )

    save_dark_graph(
        tree_counts,
        loss_values,
        "Number of Trees",
        "Test Log Loss",
        "Random Forest Log Loss",
        "loss_graph.png"
    )

    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy : {metrics_data['accuracy']}%")
    print(f"Precision: {metrics_data['precision']}%")
    print(f"Recall   : {metrics_data['recall']}%")
    print(f"F1-Score : {metrics_data['f1_score']}%")

    joblib.dump(model, "drowsiness_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    with open("metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(
        "Success! Saved model, scaler, metrics, confusion matrix, "
        "accuracy graph, and loss graph."
    )


if __name__ == "__main__":
    train_and_save_model()
