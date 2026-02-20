import joblib
import numpy as np
import re
import os
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishGuard AI Backend")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading Logic ---
# Model ka naam wahi rakhein jo aapki repository mein hai
MODEL_FILE = 'Random_Forest_Model.pkl' 

if os.path.exists(MODEL_FILE):
    try:
        # Model ko load karna
        model = joblib.load(MODEL_FILE)
        
        # Feature names mismatch fix (Warning hatane ke liye)
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_
        else:
            feature_cols = [f"f{i}" for i in range(30)] # UCI default 30 features
            
        logger.info(f"✅ Model '{MODEL_FILE}' loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Error loading model file: {e}")
        model = None
else:
    logger.error(f"❌ Critical Error: Model file '{MODEL_FILE}' not found!")
    model = None

# --- 2. Feature Extraction ---
def extract_features(url: str):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    
    # UCI 30 features logic (Sample mapping)
    # Aapka model exactly 30 features expect karta hai
    features = [0] * 30 
    features[0] = 1 if len(url) < 54 else -1
    features[1] = 1 if "@" not in url else -1
    features[6] = 1 if "https" in url[:5] else -1
    features[9] = 1 if not any(kw in url.lower() for kw in ["login", "auth", "verify"]) else -1
    
    # DataFrame banana taaki model ko 'feature names' mil sakein
    return pd.DataFrame([features], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file missing on server.")

    url = data.url.strip()
    
    # Manual Heuristics (Security Researcher logic)
    # Naye phishing domains ke liye manual block
    if any(url.endswith(tld) for tld in [".pro", ".top", ".xyz"]) or "log.php" in url:
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.99}

    try:
        # Feature extract karke prediction karna
        features_df = extract_features(url)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            # UCI dataset labels: index 0 typically means Phishing (-1)
            phishing_prob = float(probs[0])
            is_phish = phishing_prob > 0.48 # Strict threshold for safety
        else:
            prediction = model.predict(features_df)[0]
            is_phish = True if prediction == -1 else False
            phishing_prob = 1.0 if is_phish else 0.0

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2)
        }
    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}")
        return {"prediction": "error", "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
