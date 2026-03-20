import pickle
import os
import numpy as np

class SecureScanner:
    def __init__(self):
        print("🛡️ Booting Aegis Secure Scanner...")
        self.vectorizer = None
        self.classifier = None
        self.feature_names = None
        self.importances = None
        self.model_type = None
        self.active_folder = None
        
        current_file_path = os.path.abspath(__file__)
        security_dir = os.path.dirname(current_file_path)
        app_dir = os.path.dirname(security_dir)
        self.base_dir = os.path.dirname(app_dir) 

    def load_model_from_folder(self, folder_name: str):
        """Hot-reloads the ML models into RAM from a specific folder."""
        model_dir = os.path.join(self.base_dir, folder_name)
        print(f"🔄 Swapping active firewall to: {folder_name}...")
        
        try:
            with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(model_dir, "classifier.pkl"), "rb") as f:
                self.classifier = pickle.load(f)
            
            self.active_folder = folder_name
            self.feature_names = np.array(self.vectorizer.get_feature_names_out())
            
            # Auto-detect Linear vs Tree
            if hasattr(self.classifier, 'coef_'):
                self.model_type = "linear"
                self.importances = self.classifier.coef_[0]
            elif hasattr(self.classifier, 'feature_importances_'):
                self.model_type = "tree"
                self.importances = self.classifier.feature_importances_
            else:
                self.importances = np.zeros(len(self.feature_names))
                
            print(f"✅ Successfully hot-loaded {self.model_type.upper()} model from {folder_name}")
            return True
        except FileNotFoundError:
            print(f"❌ ERROR: Could not find model files in {model_dir}")
            return False

    def scan(self, text: str, threshold: float = 0.30):
        # Fallback if no model is loaded
        if self.classifier is None:
            return True, 0.0, []

        text_lower = text.lower()
        
        # --- LAYER 1: SIGNATURE CHECK ---
        known_signatures = ["do anything now", "dan", "jailbreak", "dev mode", "chaosgpt"]
        for sig in known_signatures:
            if sig in text_lower:
                return False, 1.0, [sig, "signature_match"]

        # --- LAYER 2: ML MODEL ---
        vector = self.vectorizer.transform([text])
        risk_score = self.classifier.predict_proba(vector)[0][1]
        
        is_safe = risk_score < threshold
        
        triggers = []
        if not is_safe:
            nonzero_indices = vector.nonzero()[1]
            if len(nonzero_indices) > 0:
                word_scores = [(self.feature_names[i], self.importances[i]) for i in nonzero_indices]
                word_scores.sort(key=lambda x: x[1], reverse=True)
                triggers = [word for word, score in word_scores[:3] if score > 0.0]

        return is_safe, float(risk_score), triggers