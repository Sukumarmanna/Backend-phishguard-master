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

# --- 1. CSV Data Loading ---
try:
    safe_df = pd.read_csv('safe_url.csv', dtype=str) 
    safe_list = safe_df.iloc[:, 0].dropna().astype(str).str.lower().str.strip().tolist()
    SAFE_URLS_SET = set(safe_list)
    logger.info(f"✅ Loaded {len(SAFE_URLS_SET)} safe URLs.")
except Exception as e:
    logger.error(f"❌ CSV Error: {e}")
    SAFE_URLS_SET = set()

# --- 2. Model Loading ---
MODEL_FILE = 'Random_Forest_Model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    # Model ke trained features ka order check karein
    feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"f{i}" for i in range(30)]
    logger.info("✅ ML Model Loaded.")
except Exception as e:
    logger.error(f"❌ Model Load Error: {e}")
    model = None

# --- 3. Advanced Feature Extraction (UCI 30 Features) ---
def extract_features(url):
    hostname = urlparse(url).hostname or ""
    path = urlparse(url).path or ""
    
    # Notebook ke logic ke mutabiq values: -1 (Phishing), 0 (Suspicious), 1 (Legitimate)
    f = []

    # 1. IP Address
    f.append(-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 1)
    # 2. URL Length
    f.append(1 if len(url) < 54 else (0 if len(url) <= 75 else -1))
    # 3. Shortening Service
    f.append(-1 if any(s in hostname for s in ["bit.ly", "goo.gl", "t.co", "tinyurl"]) else 1)
    # 4. At (@) Symbol
    f.append(-1 if "@" in url else 1)
    # 5. Double Slash Redirect
    f.append(-1 if url.rfind("//") > 7 else 1)
    # 6. Prefix-Suffix (Dash in Domain)
    f.append(-1 if "-" in hostname else 1)
    # 7. Sub-domain count
    dots = hostname.count('.')
    f.append(1 if dots <= 2 else (0 if dots == 3 else -1))
    # 8. SSL State (HTTPS)
    f.append(1 if url.startswith('https') else -1)
    # 9. Domain Registration Length (Placeholder)
    f.append(-1 if len(hostname) < 5 else 1)
    # 10. Favicon
    f.append(-1 if ".ico" in path else 1)
    
    # 11-20: Abnormal & HTML Features (Patterns check)
    f.append(-1 if ":" in hostname and len(hostname.split(":")) > 1 else 1) # Port
    f.append(-1 if "https" in hostname else 1) # HTTPS Token in domain
    f.append(0) # Request URL (Neutral placeholder)
    f.append(0) # URL of Anchor
    f.append(0) # Links in tags
    f.append(-1 if "sfh" in url else 1) # Server Form Handler
    f.append(-1 if "mail" in url or "mailto" in url else 1) # Submitting to Email
    f.append(0) # Abnormal URL
    f.append(1 if url.count("//") <= 1 else -1) # Redirect
    f.append(1) # On Mouseover (Default safe)
    
    # 21-30: Security & Rank Features
    f.append(1) # Right Click
    f.append(1) # Iframe
    f.append(0) # Age of Domain
    f.append(0) # DNS Record
    f.append(0) # Web Traffic
    f.append(0) # Page Rank
    f.append(0) # Google Index
    f.append(0) # Statistical Report
    f.append(1) # Extra
    f.append(1) # Extra
    
    # Model input format mein convert karein
    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    hostname = urlparse(url).hostname or ""
    
    # Priority 1: Trusted Global Domains (White-list)
    if any(domain in hostname for domain in ["google.com", "github.com", "microsoft.com", "youtube.com"]):
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # Priority 2: CSV Lookup
    if url in SAFE_URLS_SET:
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # Priority 3: ML Model Prediction
    if not model:
        return {"prediction": "safe", "is_phishing": False, "error": "Model not loaded"}

    try:
        features_df = extract_features(url)
        # Random Forest ka probability score check karein
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            # UCI data mein -1 Phishing hai aur 1 Safe
            # Index 0 usually -1 (Phishing) hota hai
            phishing_prob = probs[0] 
            is_phish = phishing_prob > 0.55 # Sensitive threshold
        else:
            pred = model.predict(features_df)[0]
            is_phish = (pred == -1)
            phishing_prob = 1.0 if is_phish else 0.0

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"prediction": "safe", "is_phishing": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
