"""
SHAP Analysis for All AQI Models (24h, 48h, 72h)
→ Saves bar plots in 'images/' and CSVs in 'csv_files/'
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ MODEL FOLDERS ============
MODEL_DIRS = ["aqi_model_24h", "aqi_model_48h", "aqi_model_72h"]
SAMPLE_SIZE = 300  # adjust if too slow

# ============ OUTPUT FOLDERS ============
IMAGE_DIR = "images"
CSV_DIR = "csv_files"
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# ============ LOAD MODEL + METADATA ============
def load_model_data(model_dir):
    """Load model, scaler, and metadata from the saved folder."""
    with open(os.path.join(model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model, scaler, feature_names, metadata

# ============ SHAP ANALYSIS ============
def run_shap(model, model_dir, X_sample, feature_names):
    print(f"\n🔹 Running SHAP for {model_dir}...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    # Compute mean absolute SHAP values per feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP value|": shap_importance
    }).sort_values("Mean |SHAP value|", ascending=False)

    # Print top features
    print(f"\nTop SHAP Feature Importances for {model_dir}:")
    print(importance_df.head(15).to_string(index=False))

    # ===== Save CSV =====
    csv_path = os.path.join(CSV_DIR, f"{model_dir}_shap_values.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"📁 SHAP importance values saved to {csv_path}")

    # ===== Save Bar Plot =====
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df["Feature"], importance_df["Mean |SHAP value|"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP Value| (Feature Importance)")
    plt.title(f"{model_dir} - SHAP Feature Importance")
    plt.tight_layout()

    image_path = os.path.join(IMAGE_DIR, f"{model_dir}_shap_bar.png")
    plt.savefig(image_path, dpi=300)
    plt.close()

    print(f"🖼️ SHAP bar plot saved to {image_path}")

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("=" * 70)
    print("📊 SHAP FEATURE IMPORTANCE (Bar Plots Only) for AQI Models (24h, 48h, 72h)")
    print("=" * 70)

    for model_dir in MODEL_DIRS:
        if not os.path.exists(model_dir):
            print(f"⚠ Folder not found: {model_dir}, skipping...")
            continue

        model, scaler, feature_names, metadata = load_model_data(model_dir)

        # Generate synthetic data (replace with real test data if available)
        X_sample = np.random.rand(SAMPLE_SIZE, len(feature_names))
        X_sample = pd.DataFrame(X_sample, columns=feature_names)
        X_sample_scaled = scaler.transform(X_sample)

        run_shap(model, model_dir, X_sample_scaled, feature_names)

    print("\n🎯 All SHAP bar plots saved in 'images/' and CSVs in 'csv_files/' folders!")
