import joblib
import numpy as np
import re
import logging
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading ---
MODEL_PATH = 'Random_Forest_Model.pkl'
try:
    model = joblib.load(MODEL_PATH)
    # Model ke training features ke naam nikalna (mismatch khatam karne ke liye)
    if hasattr(model, 'feature_names_in_'):
        feature_cols = model.feature_names_in_
    else:
        feature_cols = [f"f{i}" for i in range(30)]
    logger.info(f"✅ PhishGuard AI: Model Loaded with {len(feature_cols)} features.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

TRUSTED_DOMAINS = [
    "google.com", "youtube.com", "github.com", "microsoft.com", 
    "amazon.in", "linkedin.com", "apple.com", "instagram.com", 
    "facebook.com", "twitter.com", "netflix.com", "gmail.com"
]

# Suspicious Patterns (AI se pehle manual check)
SUSPICIOUS_TLDS = [".pro", ".top", ".xyz", ".club", ".pw", ".link", ".monster"]
SUSPICIOUS_KEYWORDS = ["auth", "login", "verify", "secure", "update", "banking", "log.php", "signin"]

# --- 2. Feature Extraction with DataFrame ---
def extract_features_df(url: str):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    
    # Basic Feature Extraction logic
    features = [
        1 if len(url) < 54 else (0 if len(url) <= 75 else -1),
        1 if "@" not in url else -1,
        1 if url.find("//", 7) == -1 else -1,
        1 if "-" not in hostname else -1,
        1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1),
        1 if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else -1,
        1 if "https" in url[:5] else -1,
        1 if len(parsed_url.path.split('/')) < 5 else -1,
        -1 if "https" in hostname else 1,
        1 if not any(kw in url.lower() for kw in ["login", "verify", "auth"]) else -1
    ]
    
    # Dummy padding to match 30 features
    features += [0] * (30 - len(features))
    
    # Warning fix: DataFrame mein convert karein taaki feature names mil jayein
    df = pd.DataFrame([features], columns=feature_cols)
    return df

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    if not model:
        raise HTTPException(status_code=500, detail="ML Model not found.")

    url = data.url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path

    # Step 1: Whitelist Check
    if any(domain == hostname or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # Step 2: Strict Heuristic Overwrite (Manual Check)
    # Agar TLD khatarnak hai ya path mein 'log.php' jaisa kuch hai
    if any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS) or \
       any(kw in path for kw in ["log.php", "login.php", "auth"]):
        logger.info(f"🚩 Manual Flag: Suspicious pattern in {url}")
        return {"prediction": "phishing", "is_phishing": True, "confidence": 0.99}

    # Step 3: ML Prediction
    try:
        features_df = extract_features_df(url)
        
        # Probabilities nikalna
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_df)[0]
            # UCI dataset mein -1 phishing hota hai aur 1 legitimate
            # Probs check: Agar phishing chance 0.4 se upar hai toh risk hai
            phishing_score = float(probs[0]) 
            is_phishing = True if phishing_score > 0.45 else False # Threshold lowered for safety
        else:
            prediction = model.predict(features_df)[0]
            is_phishing = True if prediction == -1 else False
            phishing_score = 1.0 if is_phishing else 0.0

        return {
            "prediction": "phishing" if is_phishing else "safe",
            "is_phishing": bool(is_phishing),
            "confidence": round(phishing_score, 2)
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return {"prediction": "error", "is_phishing": False}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
