
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "domain_model_a.joblib"
OUTPUT_DIR = BASE_DIR / "models"

# Load trained model
saved = joblib.load(MODEL_FILE)
model = saved["model"]
features = saved["features"]

# Extract feature importance
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance.to_string(index=False))

# Save results
importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# Plot
plt.figure(figsize=(10, 7))
plt.barh(
    importance["Feature"][::-1],
    importance["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed!")
