import os
import joblib

# Absolute-safe way to load model regardless of where script is run from
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "xgb_adr_model.pkl")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print("✅ Model loaded successfully.")
