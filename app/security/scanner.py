import pickle
import os
import time
import requests
import numpy as np
from dotenv import load_dotenv
from gradio_client import Client
import re

# 1. Force python to find the exact path to the .env file locally
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
env_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=env_path)

def sanitize_prompt(text: str) -> str:
    """
    LAYER 0: Pre-processing & Text Sanitization
    Defeats Token Smuggling and Invisible Character obfuscation.
    """
    # 1. Strip invisible zero-width characters
    # Hackers use these to split tokens without the human eye noticing
    clean_text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    
    # 2. Defeat single-letter token smuggling (e.g., s-y-s-t-e-m, p r o m p t, h.a.c.k)
    # This regex hunts for sequences of 4 or more isolated letters separated by punctuation/spaces.
    def squash_word(match):
        # Remove all non-alphanumeric characters from the matched smuggled word
        return re.sub(r'[^a-zA-Z0-9]', '', match.group(0))
        
    # The regex: exactly one letter, followed by exactly one separator, repeated 3+ times, ending in a letter.
    # It ignores normal words, but catches "s-y-s-t-e-m".
    clean_text = re.sub(r'(?:[a-zA-Z][-_.\s]){3,}[a-zA-Z]', squash_word, clean_text)
    
    return clean_text

class SecureScanner:
    def __init__(self):
        print("🛡️ Booting Aegis Secure Scanner (Microservice Mode)...")
        self.vectorizer = None
        self.classifier = None
        
        self.feature_names = None
        self.importances = None
        self.active_folder = None
        
        # --- THE LATENCY FIX: Placeholder for the global Hugging Face connection ---
        self.dl_client = None 
        
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
            print("✅ Cascade Mode active. Connecting to Hugging Face Layer 2 API...")
            # --- THE LATENCY FIX: Initialize the connection exactly ONCE during server boot! ---
            try:
                self.dl_client = Client("Mukta9904/aegis-dl-api")
                print("✅ Successfully connected to Cloud DL Layer.")
            except Exception as e:
                print(f"⚠️ Failed to connect to HF space: {e}")
                
        return True

    def scan(self, text: str, threshold: float = 0.45):
        start_time = time.perf_counter() 
        
        # --- SANITIZATION ---
        # Clean the text to defeat token smuggling before any ML happens
        original_text = text 
        text = sanitize_prompt(text)
        
        # If the vectorizer or classifier are not loaded, return a safe default
        if self.classifier is None:
            return True, 0.0, [], "None", 0.0

        text_lower = text.lower()
        
        # --- LAYER 0a: HARD SIGNATURE BLOCK ---
        # These are undisputed malicious commands. Block immediately to save compute.
        #hard_signatures = []
        #for sig in hard_signatures:
            #if sig in text_lower:
                #latency = round((time.perf_counter() - start_time) * 1000, 2)
                #return False, 1.0, [sig, "hard_signature_match"], "Layer 0 (Signature)", latency

        # --- LAYER 0b: THE ROLEPLAY TRIPWIRE (FIX 2 - Conditional Escalation) ---
        
        # These indicate complex framing. We don't block them, we FORCE a Deep Learning scan.
        roleplay_tripwires = [
            "you are a", "act as a", "ignore all previous", 
            "forget your previous", "assume the persona", 
            "you are now", "system override", "do anything now", "dan", "jailbreak", "chaosgpt"
        ]
        
        active_tripwire = next((t for t in roleplay_tripwires if t in text_lower), None)
        force_deep_scan = active_tripwire is not None

        # --- FAST ML (LAYER 1) ---
        vector = self.vectorizer.transform([text])
        risk_score = float(self.classifier.predict_proba(vector)[0][1])
        
        triggers = []
        nonzero_indices = vector.nonzero()[1]
        if len(nonzero_indices) > 0:
            word_scores = [(self.feature_names[i], self.importances[i]) for i in nonzero_indices]
            word_scores.sort(key=lambda x: x[1], reverse=True)
            triggers = [word for word, score in word_scores[:3] if score > 0.0]
            
        # Add the tripwire phrase to the triggers so it shows up in your admin dashboard
        if force_deep_scan and active_tripwire not in triggers:
            triggers.append(active_tripwire)

        # --- CASCADE ROUTING (CALLING HUGGING FACE DL LAYER) ---
        if self.active_folder == "cascade":
            
            # --- DYNAMIC LENGTH THRESHOLDING (FIX 3) ---
            # If the prompt is over 500 characters, widen the net to catch buried attacks
            prompt_length = len(original_text)
            cascade_lower_bound = 0.05 if prompt_length > 500 else 0.15
            
            # ROUTE TO DL IF: Tripwire triggered OR Score is in the Grey Area
            if force_deep_scan or (cascade_lower_bound <= risk_score <= 0.85):
                
                # Log exactly why it was escalated to the cloud
                reason = f"Tripwire ['{active_tripwire}']" if force_deep_scan else f"Grey Area ({risk_score:.2f} | Len: {prompt_length})"
                print(f"🟡 {reason}. Routing to Layer 2 DL...")
                
                try:
                    if self.dl_client is None:
                        raise Exception("Hugging Face Client was not initialized properly at startup.")
                        
                    dl_risk_score, dl_triggers = self.dl_client.predict(
                        text=original_text, # Passing the original text so DistilBERT sees full context
                        api_name="/analyze_prompt"
                    )
                    
                    extracted_triggers = list(dl_triggers.keys())
                    if force_deep_scan: 
                        extracted_triggers.append("roleplay_framing")
                        
                    # Using the tighter 0.40 threshold to catch subtle roleplays
                    is_safe = dl_risk_score < 0.35 
                    latency = round((time.perf_counter() - start_time) * 1000, 2)
                    
                    return is_safe, dl_risk_score, extracted_triggers, "Layer 2 (LP + SHAP)", latency
                    
                except Exception as e:
                    print(f"⚠️ Layer 2 API Failed: {e}. Falling back to Layer 1!")
                    is_safe = risk_score < 0.50
                    latency = round((time.perf_counter() - start_time) * 1000, 2)
                    return is_safe, risk_score, triggers, "Layer 1", latency
            else:
                is_safe = risk_score < 0.50 
                latency = round((time.perf_counter() - start_time) * 1000, 2)
                return is_safe, risk_score, triggers, "Layer 1", latency

        # --- STANDARD ML LOGIC ---
        else:
            is_safe = risk_score < threshold
            latency = round((time.perf_counter() - start_time) * 1000, 2)
            if is_safe: triggers = [] 
            return is_safe, risk_score, triggers, "Layer 1", latency