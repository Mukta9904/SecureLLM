import pickle
import os
import time
import requests
import numpy as np
from dotenv import load_dotenv

# 1. Force python to find the exact path to the .env file locally
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
env_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=env_path)

# 2. Grab the URL from the system. 
COLAB_API_URL = os.getenv("COLAB_API_URL")

# 3. IF it is still None (because we forgot to set it), force a safe default so it doesn't crash!
if not COLAB_API_URL:
    COLAB_API_URL = "http://localhost:8000/scan_layer2"

class SecureScanner:
    def __init__(self):
        print("🛡️ Booting Aegis Secure Scanner (Microservice Mode)...")
        print(f"🔗 Layer 2 Endpoint set to: {COLAB_API_URL}")
        self.vectorizer = None
        self.classifier = None
        
        self.feature_names = None
        self.importances = None
        self.active_folder = None
        
        current_file_path = os.path.abspath(__file__)
        security_dir = os.path.dirname(current_file_path)
        app_dir = os.path.dirname(security_dir)
        self.base_dir = os.path.dirname(app_dir) 

    def load_model_from_folder(self, folder_name: str):
        print(f"🔄 Swapping active firewall to: {folder_name}...")
        self.active_folder = folder_name
        
        # ALWAYS LOAD LAYER 1 (Logistic Regression)
        lr_dir = os.path.join(self.base_dir, "LR_models" if folder_name == "cascade" else folder_name)
        
        try:
            with open(os.path.join(lr_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(lr_dir, "classifier.pkl"), "rb") as f:
                self.classifier = pickle.load(f)
            
            self.feature_names = np.array(self.vectorizer.get_feature_names_out())
            self.importances = self.classifier.coef_[0] if hasattr(self.classifier, 'coef_') else self.classifier.feature_importances_
            print(f"✅ Loaded Layer 1 (Fast ML) from {lr_dir}")
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find Layer 1 files in {lr_dir}")
            return False

        if folder_name == "cascade":
            print("✅ Cascade Mode active. Layer 2 requests will be routed to Colab GPU.")
            
        return True

    def scan(self, text: str, threshold: float = 0.45):
        start_time = time.perf_counter() 
        
        if self.classifier is None:
            return True, 0.0, [], "None", 0.0

        text_lower = text.lower()
        
        # --- LAYER 0: SIGNATURE CHECK ---
        known_signatures = ["do anything now", "dan", "jailbreak", "dev mode", "chaosgpt"]
        for sig in known_signatures:
            if sig in text_lower:
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return False, 1.0, [sig, "signature_match"], "Layer 0 (Signature)", latency

        # --- LAYER 1: FAST ML ---
        vector = self.vectorizer.transform([text])
        risk_score = float(self.classifier.predict_proba(vector)[0][1])
        
        triggers = []
        nonzero_indices = vector.nonzero()[1]
        if len(nonzero_indices) > 0:
            word_scores = [(self.feature_names[i], self.importances[i]) for i in nonzero_indices]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            triggers = [word for word, score in word_scores[:3] if score > 0.0]

        # --- CASCADE ROUTING (CALLING COLAB) ---
        if self.active_folder == "cascade":
            if 0.15 <= risk_score <= 0.85:
                # 🚀 The text is in the grey area. Send to Colab GPU!
                try:
                    response = requests.post(COLAB_API_URL, json={"text": text}, timeout=3.0)
                    response.raise_for_status()
                    dl_data = response.json()
                    
                    is_safe = dl_data["is_safe"]
                    dl_risk_score = dl_data["risk_score"]
                    latency = round((time.perf_counter() - start_time) * 1000, 2)
                    
                    return is_safe, dl_risk_score, [], "Layer 2 (Colab DistilBERT)", latency
                    
                except Exception as e:
                    print(f"⚠️ Colab API Failed/Timeout: {e}. Falling back to Layer 1!")
                    # If Colab is asleep or dead, gracefully fall back to Layer 1 decision
                    is_safe = risk_score < 0.50
                    latency = round((time.perf_counter() - start_time) * 1000, 2)
                    return is_safe, risk_score, triggers, "Layer 1 (Fallback Mode)", latency
            else:
                is_safe = risk_score < 0.50 
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return is_safe, risk_score, triggers, "Layer 1 (Logistic Regression)", latency

        # --- STANDARD ML LOGIC ---
        else:
            is_safe = risk_score < threshold
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            if is_safe: triggers = [] 
            return is_safe, risk_score, triggers, "Layer 1 (Logistic Regression)", latency