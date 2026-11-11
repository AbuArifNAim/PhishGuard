
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_dataset.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Separate features and target
X = df.drop(columns=["URL", "ClassLabel"])
y = df["ClassLabel"]

# Two experiments
experiments = {
    "model_a_all_features": X,
    "model_b_without_https": X.drop(columns=["https_flag"])
}

# Same train-test split for fair comparison
X_train_idx, X_test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42,
    stratify=y
)

y_train = y.loc[X_train_idx]
y_test = y.loc[X_test_idx]

for model_name, features in experiments.items():

    print("\n" + "=" * 55)
    print("Training:", model_name)
    print("=" * 55)

    X_train = features.loc[X_train_idx]
    X_test = features.loc[X_test_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("\nAccuracy:",
          round(accuracy_score(y_test, predictions), 4))

    print("Precision:",
          round(precision_score(y_test, predictions), 4))

    print("Recall:",
          round(recall_score(y_test, predictions), 4))

    print("F1 Score:",
          round(f1_score(y_test, predictions), 4))

    print("ROC-AUC:",
          round(roc_auc_score(y_test, probabilities), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Phishing", "Legitimate"]
    ))

    # Save model and feature names
    model_data = {
        "model": model,
        "features": list(features.columns),
        "positive_class": 1,
        "negative_class": 0
    }

    output_file = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model_data, output_file)

    print("Saved:", output_file)

print("\nAll experiments completed!")
