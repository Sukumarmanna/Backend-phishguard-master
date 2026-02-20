import joblib
import numpy as np
import re
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishGuard AI Backend")

# CORS Settings - Extension se connect karne ke liye zaroori hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Model Loading Logic ---
MODEL_PATH = 'Random_Forest_Model.pkl'
FALLBACK_MODEL = 'Decision_Tree_Model.pkl'

try:
    # Model load karne ke liye joblib.load use karna must hai
    model = joblib.load(MODEL_PATH)
    logger.info(f"✅ PhishGuard AI: {MODEL_PATH} Loaded successfully.")
except Exception as e:
    try:
        model = joblib.load(FALLBACK_MODEL)
        logger.info(f"⚠️ Falling back to {FALLBACK_MODEL}.")
    except Exception as e:
        logger.error(f"❌ Failed to load any model: {e}")
        model = None

# --- 2. Advanced Feature Extraction ---
# Note: UCI Dataset requires specific features. Ye function basic indicators capture karta hai.
def extract_features(url: str):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""
    
    # Matching your notebook's expected input structure
    features = [
        url.count('.'),
        len(url),
        url.count('-'),
        1 if '@' in url else 0,
        1 if "//" in url[8:] else 0, # Double slash check
        hostname.count('.'),
        1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0, # IP address check
        1 if "https" in url[:5] else 0,
        len(path.split('/')),
        sum(c.isdigit() for c in url)
    ]
    
    # IMPORTANT: Agar aapka model 30 features (UCI) mang raha hai, 
    # toh aapko model training ke waqt use kiye gaye columns yahan replicate karne honge.
    # Filhal hum dummy padding de rahe hain agar model shape mismatch kare.
    expected_features = getattr(model, "n_features_in_", 10)
    if len(features) < expected_features:
        features += [0] * (expected_features - len(features))
        
    return np.array(features).reshape(1, -1)

# --- 3. Whitelist & API Endpoints ---
TRUSTED_DOMAINS = ["google.com", "github.com", "microsoft.com", "amazon.in", "linkedin.com", "apple.com"]

class URLInput(BaseModel):
    url: str

@app.post("/predict")
async def predict(data: URLInput):
    if not model:
        raise HTTPException(status_code=500, detail="ML Model not initialized.")

    url = data.url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    hostname = urlparse(url).hostname or ""

    # Step 1: Whitelist Check
    if any(domain == hostname or hostname.endswith('.' + domain) for domain in TRUSTED_DOMAINS):
        return {"prediction": "safe", "is_phishing": False, "confidence": 1.0}

    # Step 2: ML Prediction
    try:
        features = extract_features(url)
        
        # Prediction
        prediction = model.predict(features)[0]
        
        # Confidence logic
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            phishing_chance = float(probabilities[1])
        else:
            phishing_chance = 1.0 if prediction == 1 else 0.0

        # Phishing detection threshold (Adjustable)
        is_phishing = phishing_chance > 0.75 
        
        return {
            "prediction": "phishing" if is_phishing else "safe",
            "is_phishing": bool(is_phishing),
            "confidence": round(phishing_chance, 2)
        }
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return {"prediction": "error", "is_phishing": False, "detail": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)