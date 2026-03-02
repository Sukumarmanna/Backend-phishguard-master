import joblib
import numpy as np
import re
import os
import pandas as pd
import whois
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading ---
MODEL_FILE = 'Random_Forest_Model.pkl'
try:
    model = joblib.load(MODEL_FILE)
    feature_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [f"f{i}" for i in range(30)]
except Exception as e:
    model = None

# --- 2. Helper: Domain Age Logic ---
def get_domain_age(domain):
    try:
        w = whois.whois(domain)
        start_date = w.creation_date
        if isinstance(start_date, list): start_date = start_date[0]
        if start_date:
            age_months = (datetime.now() - start_date).days // 30
            return 1 if age_months >= 6 else -1 # UCI logic: >=6 months is safe
        return -1
    except:
        return -1

# --- 3. Complete 30-Feature Extraction ---
def extract_30_features(url):
    hostname = urlparse(url).hostname or ""
    path = urlparse(url).path or ""
    f = []

    # 1-10: Address Bar Features
    f.append(-1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 1) # IP
    f.append(1 if len(url) < 54 else (0 if len(url) <= 75 else -1))      # URL Length
    f.append(-1 if any(s in hostname for s in ["bit.ly", "t.co"]) else 1) # Shortener
    f.append(-1 if "@" in url else 1)                                     # @ symbol
    f.append(-1 if url.rfind("//") > 7 else 1)                           # Double slash
    f.append(-1 if "-" in hostname else 1)                               # Dash in domain
    f.append(1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1)) # Subdomain
    f.append(1 if url.startswith('https') else -1)                       # SSL State
    f.append(1) # 9. Domain Registration (Static filler)
    f.append(-1 if ".ico" in path else 1)                                # Favicon

    # 11-20: Abnormal Features
    f.append(-1 if ":" in hostname else 1)                               # Port
    f.append(-1 if "https" in hostname else 1)                           # HTTPS Token
    f.append(0); f.append(0); f.append(0)                                # 13,14,15: Request/Anchor/Tags
    f.append(-1 if "sfh" in url else 1)                                  # SFH
    f.append(-1 if "mail" in url else 1)                                 # Email
    f.append(0)                                                          # 18: Abnormal URL
    f.append(1 if url.count("//") <= 1 else -1)                          # Redirect
    f.append(1)                                                          # 20: MouseOver

    # 21-30: Domain & Statistical Features
    f.append(1) # 21: Right Click
    f.append(1) # 22: IFrame
    f.append(get_domain_age(hostname))                                   # 23: Age of Domain (REAL DATA)
    f.append(1) # 24: DNS Record
    f.append(0) # 25: Web Traffic
    f.append(0) # 26: Page Rank
    f.append(0) # 27: Google Index
    f.append(0) # 28: Links pointing to page
    f.append(0) # 29: Statistical Report
    f.append(1) # 30: Target (Filler)

    return pd.DataFrame([f], columns=feature_cols)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    url = data.url.lower().strip()
    try:
        features_df = extract_30_features(url)
        
        # ML Prediction
        probs = model.predict_proba(features_df)[0]
        phishing_prob = float(probs[0]) 
        
        # Override: Agar domain age -1 (naya) hai, toh sensitivity badhao
        is_phish = phishing_prob > 0.40 if features_df.iloc[0, 22] == -1 else phishing_prob > 0.60

        return {
            "prediction": "phishing" if is_phish else "safe",
            "is_phishing": bool(is_phish),
            "confidence": round(phishing_prob, 2),
            "age_feature": int(features_df.iloc[0, 22])
        }
    except Exception as e:
        return {"prediction": "safe", "is_phishing": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
