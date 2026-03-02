import joblib
import numpy as np
import re
import os
import pandas as pd
import whois
import logging
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Logging setup taaki Render logs mein error dikhe
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading (Safe Load) ---
MODEL_FILE = 'Random_Forest_Model.pkl'
model = None
feature_cols = [f"f{i}" for i in range(30)]

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_
        logger.info("✅ ML Model Loaded Successfully")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
else:
    logger.warning("⚠️ Model file not found! Using fallback logic.")

# --- 2. Helper: Domain Age Logic (Feature 23) ---
def get_domain_age(domain):
    try:
        # Timeout add kiya hai taaki request hang na ho
        w = whois.whois(domain)
        start_date = w.creation_date
        if isinstance(start_date, list): 
            start_date = start_date[0]
        
        if start_date and isinstance(start_date, datetime):
            age_months = (datetime.now() - start_date).days // 30
            return 1 if age_months >= 6 else -1 # UCI: 6+ months is safe
        return -1
    except Exception:
        return -1 # Default to suspicious if WHOIS fails

# --- 3. 30-Feature Extraction (UCI Standard) ---
def extract_30_features(url):
    hostname = urlparse(url).hostname or ""
    path = urlparse(url).path or ""
    f = []

    # 1-10: Address Bar
    f.append(-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 1) # IP
    f.append(1 if len(url) < 54 else (0 if len(url) <= 75 else -1))      # Length
    f.append(-1 if any(s in hostname for s in ["bit.ly", "t.co", "goo.gl"]) else 1) # Shortener
    f.append(-1 if "@" in url else 1)                                     # @
    f.append(-1 if url.rfind("//") > 7 else 1)                           # //
    f.append(-1 if "-" in hostname else 1)                               # Dash
    f.append(1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1)) # Subdomain
    f.append(1 if url.startswith('https') else -1)                       # SSL
    f.append(1) # Registration
    f.append(-1 if ".ico" in path else 1)                                # Favicon

    # 11-20: Abnormal
    f.append(-1 if ":" in hostname else 1)                               # Port
    f.append(-1 if "https" in hostname else 1)                           # HTTPS Token
    f.append(0); f.append(0); f.append(0)                                # Static
    f.append(-1 if "sfh" in url else 1)                                  # SFH
    f.append(-1 if "mail" in url else 1)                                 # Email
    f.append(0)                                                          # Abnormal
    f.append(1 if url.count("//") <= 1 else -1)                          # Redirect
    f.append(1)                                                          # MouseOver

    # 21-30: Domain & Statistics
    f.append(1) # Right Click
    f.append(1) # IFrame
    f.append(get_domain_age(hostname))                                   # 23: Age of Domain
    f.append(1) # DNS
    f.append(0); f.append(0); f.append(0); f.append(0); f.append(0)       # Static
    f.append(1) # Target

    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    try:
        features_df = extract_30_features(url)
        age_val = int(features_df.iloc[0, 22])
        
        if model:
            probs = model.predict_proba(features_df)[0]
            phishing_prob = float(probs[0]) # Prob of class -1 (Phishing)
            
            # Sensitivity Adjustment
            # Agar domain naya hai (-1), toh 40% shak par bhi phishing bolo
            threshold = 0.40 if age_val == -1 else 0.60
            is_phish = phishing_prob > threshold
        else:
            # Fallback logic agar model load na ho
            is_phish = True if age_val == -1 else False
            phishing_prob = 0.5

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2),
            "age_feature": age_val
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"prediction": "safe", "is_phishing": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Render $PORT environment variable use karta hai
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
