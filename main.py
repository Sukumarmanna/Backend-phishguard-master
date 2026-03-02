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
        logger.info("✅ ML Model Loaded Successfully")
    except Exception as e:
        logger.error(f"❌ Model Error: {e}")

# --- 2. Helper: Enhanced Domain Age ---
def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        start_date = w.creation_date
        if isinstance(start_date, list): start_date = start_date[0]
        if start_date and isinstance(start_date, datetime):
            age_months = (datetime.now() - start_date).days // 30
            # UCI standard: Age >= 6 months is legitimate (1), else phishing (-1)
            return 1 if age_months >= 6 else -1
        return -1
    except:
        return -1

# --- 3. Full 30-Feature Extraction ---
def extract_30_features(url):
    hostname = urlparse(url).hostname or ""
    path = urlparse(url).path or ""
    f = []

    # Address Bar Features (1-10)
    f.append(-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 1) # IP
    f.append(1 if len(url) < 54 else (0 if len(url) <= 75 else -1)) # Length
    f.append(-1 if any(s in hostname for s in ["bit.ly", "t.co", "goo.gl"]) else 1) # Shortener
    f.append(-1 if "@" in url else 1) # @
    f.append(-1 if url.rfind("//") > 7 else 1) # //
    f.append(-1 if "-" in hostname else 1) # Dash
    f.append(1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1)) # Subdomain
    f.append(1 if url.startswith('https') else -1) # SSL
    f.append(1) # Registration
    f.append(-1 if ".ico" in path else 1) # Favicon

    # Abnormal & Technical Features (11-20)
    f.append(-1 if ":" in hostname else 1) # Port
    f.append(-1 if "https" in hostname else 1) # HTTPS Token in Host
    f.append(0); f.append(0); f.append(0) # Static placeholders
    # SFH: Path mein login/verify patterns check karna
    f.append(-1 if any(kw in path.lower() for kw in ["login", "verify", "secure", "bank"]) else 1)
    f.append(-1 if "mail" in url else 1) # Email
    f.append(0) # Abnormal URL
    f.append(1 if url.count("//") <= 1 else -1) # Redirect
    f.append(1) # MouseOver

    # Domain & Statistical (21-30)
    f.append(1); f.append(1)
    f.append(get_domain_age(hostname)) # 23: Age of Domain (REAL-TIME)
    f.append(1); f.append(0); f.append(0); f.append(0); f.append(0); f.append(0); f.append(1)

    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    hostname = urlparse(url).hostname or ""
    
    # --- LAYER 1: Hard Whitelist ---
    trusted = ["google.com", "github.com", "microsoft.com", "youtube.com", "wikipedia.org"]
    if any(d in hostname for d in trusted):
        return {"prediction": "safe", "is_phishing": False, "confidence": 0.0}

    # --- LAYER 2: Advanced Heuristics (Aggressive Detection) ---
    # Phishing platforms aur keywords ka combination
    suspicious_hosts = ['webflow.io', 'netlify.app', 'sbs', 'xyz', 'trycloudflare.com', 'lhr.life']
    critical_keywords = ['ledger', 'google', 'microsoft', 'bank', 'login', 'verify', 'carousell', 'crypto']
    
    # Logic A: Agar suspicious host par brand name hai
    if any(sh in hostname for sh in suspicious_hosts):
        if any(ck in url for ck in critical_keywords):
            return {"prediction": "phishing", "is_phishing": True, "confidence": 0.98, "reason": "Phishing Host Pattern"}
    
    # Logic B: Brand Spoofing (e.g., google-security.net)
    if "google" in url and "google.com" not in hostname:
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.95, "reason": "Brand Spoofing"}

    # --- LAYER 3: ML Model Prediction ---
    try:
        features_df = extract_30_features(url)
        age_val = int(features_df.iloc[0, 22]) # Feature 23 (Age)
        
        if model:
            probs = model.predict_proba(features_df)[0]
            phishing_prob = float(probs[0]) # Prob of class -1
            
            # SENSITIVITY OVERRIDE: 
            # Agar domain naya hai (-1), toh 30% shak par bhi block karein
            threshold = 0.30 if age_val == -1 else 0.50
            is_phish = phishing_prob > threshold
        else:
            is_phish = (age_val == -1)
            phishing_prob = 0.5

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2),
            "age_feature": age_val,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"prediction": "safe", "is_phishing": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
