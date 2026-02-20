import joblib
import numpy as np
import re
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishGuard AI Backend")

# CORS Settings - Extension connection ke liye must hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading Logic ---
MODEL_PATH = 'Random_Forest_Model.pkl'

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ PhishGuard AI: {MODEL_PATH} Loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# --- 2. Whitelist & Trusted Domains ---
# YouTube aur social media ko yahan add kiya hai taaki False Positives na aayein
TRUSTED_DOMAINS = [
    "google.com", "youtube.com", "github.com", "microsoft.com", 
    "amazon.in", "linkedin.com", "apple.com", "instagram.com", 
    "facebook.com", "twitter.com", "netflix.com"
]

# --- 3. UCI-Based Feature Extraction ---
def extract_features(url: str):
    """
    Note: UCI Dataset 30 features use karta hai. 
    Ye simplified function model shape (30,) ko match karne ke liye padding karta hai.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    
    # Basic Feature Engineering (Replicating UCI logic)
    features = [
        1 if len(url) < 54 else (0 if len(url) <= 75 else -1),  # URL Length
        1 if "@" not in url else -1,                          # Having @ symbol
        1 if url.find("//", 7) == -1 else -1,                 # Double slash redirect
        1 if "-" not in hostname else -1,                     # Prefix-Suffix in domain
        1 if hostname.count('.') <= 2 else (0 if hostname.count('.') == 3 else -1), # Dots in domain
        1 if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else -1, # IP Address check
        1 if "https" in url[:5] else -1,                      # HTTPS check
        1 if len(parsed_url.path.split('/')) < 5 else -1,     # URL Depth
        -1 if "https" in hostname else 1,                     # HTTPS in domain part
        1 if "login" not in url.lower() else -1               # Login/Sign-in keywords
    ]
    
    # Model 30 features expect karta hai
    # Baki 20 features ko dummy values se fill karte hain
    expected_features = 30 
    if len(features) < expected_features:
        features += [1] * (expected_features - len(features))
        
    return np.array(features).reshape(1, -1)

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    if not model:
        raise HTTPException(status_code=500, detail="ML Model not found on server.")

    url = data.url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    hostname = urlparse(url).hostname or ""

    # Step 1: Whitelist Check (YouTube Fix)
    if any(domain == hostname or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        logger.info(f"🟢 Whitelist Hit: {hostname}")
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # Step 2: ML Prediction
    try:
        features = extract_features(url)
        prediction = model.predict(features)[0]
        
        # Random Forest confidence score
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            # UCI labels usually are -1 (phishing) and 1 (legitimate)
            # Probability index depends on how model was trained
            phishing_chance = float(probabilities[0]) if prediction == -1 else (1 - float(probabilities[1]))
        else:
            phishing_chance = 1.0 if prediction == -1 else 0.0

        # Thresholding
        is_phishing = True if prediction == -1 else False
        
        return {
            "prediction": "phishing" if is_phishing else "safe",
            "is_phishing": bool(is_phishing),
            "confidence": round(phishing_chance, 2)
        }
    except Exception as e:
        logger.error(f"❌ Prediction Error: {e}")
        return {"prediction": "error", "is_phishing": False, "detail": "Analysis failed"}

# --- 4. Render Dynamic Port Handling ---
if __name__ == "__main__":
    import uvicorn
    # Render $PORT environment variable use karta hai
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
