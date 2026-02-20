import joblib
import numpy as np
import re
import os
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CSV Data Loading (Robust Fix) ---
try:
    # safe_url.csv load karte waqt sabhi values ko string force karein
    safe_df = pd.read_csv('safe_url.csv', dtype=str) 
    
    # Column 0 se URLs nikalna aur null/empty values hatana
    safe_list = safe_df.iloc[:, 0].dropna().astype(str).str.lower().str.strip().tolist()
    
    SAFE_URLS_SET = set(safe_list)
    logger.info(f"✅ Loaded {len(SAFE_URLS_SET)} safe URLs from CSV.")
except Exception as e:
    logger.error(f"❌ CSV Load Error: {e}")
    SAFE_URLS_SET = set()
# --- 2. Model Loading ---
MODEL_FILE = 'Random_Forest_Model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"f{i}" for i in range(30)]
    logger.info("✅ ML Model Loaded.")
except Exception as e:
    logger.error(f"❌ Model Load Error: {e}")
    model = None

# --- 3. Feature Extraction ---
def extract_features(url):
    # UCI 30 features logic
    f = [1] * 30 # Default 1 (Safe)
    hostname = urlparse(url).hostname or ""
    f[0] = 1 if len(url) < 54 else (-1 if len(url) > 75 else 0)
    f[6] = 1 if url.startswith('https') else -1
    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    
    # STEP 1: Direct CSV Lookup (First Priority)
    # Agar URL aapki safe_url.csv mein hai, toh ML ki zaroorat hi nahi
    if url in SAFE_URLS_SET or any(domain in url for domain in ["google.com", "youtube.com"]):
        logger.info(f"🟢 Safe CSV Match: {url}")
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # STEP 2: Manual Pattern Check (Security Researcher Logic)
    if ".pro" in url or "log.php" in url:
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.99}

    # STEP 3: ML Model Prediction
    if not model:
        return {"prediction": "safe", "is_phishing": False}

    try:
        features_df = extract_features(url)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            phishing_score = float(probs[0])
            # Threshold balance: Agar 70% shak hai tabhi Danger
            is_phish = phishing_score > 0.70 
        else:
            is_phish = model.predict(features_df)[0] == -1
            phishing_score = 1.0 if is_phish else 0.0

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_score, 2)
        }
    except Exception as e:
        return {"prediction": "safe", "is_phishing": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
