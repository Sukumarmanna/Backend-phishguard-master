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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading ---
MODEL_FILE = 'Random_Forest_Model.pkl'
model = None
feature_cols = [f"f{i}" for i in range(30)]

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_
        logger.info("✅ ML Model Loaded")
    except Exception as e:
        logger.error(f"❌ Model Error: {e}")

# --- 2. Helper: Domain Age (Feature 23) ---
def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        start_date = w.creation_date
        if isinstance(start_date, list): start_date = start_date[0]
        if start_date and isinstance(start_date, datetime):
            age_months = (datetime.now() - start_date).days // 30
            return 1 if age_months >= 6 else -1
        return -1
    except:
        return -1

# --- 3. 30-Feature Extraction ---
def extract_30_features(url):
    hostname = urlparse(url).hostname or ""
    path = urlparse(url).path or ""
    f = []

    # Address Bar Features
    f.append(-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 1) # IP
    f.append(1 if len(url) < 54 else (0 if len(url) <= 75 else -1)) # Length
    f.append(-1 if any(s in hostname for s in ["bit.ly", "t.co", "goo.gl"]) else 1) # Shortener
    f.append(-1 if "@" in url else 1) # @
    f.append(-1 if url.rfind("//") > 7 else 1) # //
    f.append(-1 if "-" in hostname else 1) # Dash
    f.append(1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1)) # Subdomain
    f.append(1 if url.startswith('https') else -1) # SSL
    f.append(1); f.append(-1 if ".ico" in path else 1)

    # Abnormal & Static Features
    f.append(-1 if ":" in hostname else 1); f.append(-1 if "https" in hostname else 1)
    for _ in range(3): f.append(0) # Static placeholders
    f.append(-1 if any(kw in path.lower() for kw in ["login", "verify", "secure"]) else 1) # SFH Logic
    f.append(-1 if "mail" in url else 1); f.append(0); f.append(1 if url.count("//") <= 1 else -1); f.append(1)

    # Domain & Stats
    f.append(1); f.append(1)
    f.append(get_domain_age(hostname)) # 23: Age (Critical)
    f.append(1); f.append(0); f.append(0); f.append(0); f.append(0); f.append(0); f.append(1)

    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    hostname = urlparse(url).hostname or ""
    
    # --- LAYER 1: Hard Whitelist (Trusted Global Domains) ---
    if any(d in hostname for d in ["google.com", "github.com", "microsoft.com", "youtube.com"]):
        return {"prediction": "safe", "is_phishing": False, "confidence": 0.0}

    # --- LAYER 2: Suspicious Patterns (Brand Protection) ---
    # Agar URL mein 'google' hai par domain 'google.com' nahi hai
    if "google" in url and "google.com" not in hostname:
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.95, "reason": "Brand Spoofing"}
    
    # Suspicious TLDs check
    if any(hostname.endswith(ext) for ext in [".sbs", ".xyz", ".online", ".pro", ".cloud"]):
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.88, "reason": "Suspicious TLD"}

    # --- LAYER 3: ML Model Prediction ---
    try:
        features_df = extract_30_features(url)
        age_val = int(features_df.iloc[0, 22]) # Feature 23
        
        if model:
            probs = model.predict_proba(features_df)[0]
            phishing_prob = float(probs[0])
            
            # AGGRESSIVE SENSITIVITY: 
            # Agar domain naya hai (-1), toh sirf 35% shak par bhi DANGER bol do
            threshold = 0.35 if age_val == -1 else 0.55
            is_phish = phishing_prob > threshold
        else:
            is_phish = (age_val == -1)
            phishing_prob = 0.5

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2),
            "age_feature": age_val
        }
    except Exception as e:
        return {"prediction": "safe", "is_phishing": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
