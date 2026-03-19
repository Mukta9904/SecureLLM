import time
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_DIR = "new_models"

print("🛡️ Booting Aegis Advanced Evaluation Suite...\n")

try:
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "rb") as f:
        classifier = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "test_data.pkl"), "rb") as f:
        test_vault = pickle.load(f)
        X_test = test_vault["X_test"]
        y_true = test_vault["y_test"]
    print("✅ Models and Test Vault loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Missing files. Run train_model.py first.")
    exit()

print(f"⚖️ Evaluating against {len(X_test)} pristine, unseen prompts...")

start_vec_time = time.perf_counter()
X_test_vectors = vectorizer.transform(X_test)

THRESHOLD = 0.30 
y_probs = classifier.predict_proba(X_test_vectors)[:, 1]
y_pred = [1 if prob >= THRESHOLD else 0 for prob in y_probs]
end_time = time.perf_counter()

avg_latency_ms = ((end_time - start_vec_time) * 1000) / len(X_test)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n" + "="*45)
print("📊 AEGIS PRODUCTION PERFORMANCE REPORT")
print("="*45)
print(f"🎯 Accuracy:  {accuracy * 100:.2f}%")
print(f"🛡️ Precision: {precision * 100:.2f}%")
print(f"🔍 Recall:    {recall * 100:.2f}%")
print(f"⚖️ F1-Score:  {f1 * 100:.2f}%")
print("-" * 45)
print(f"⏱️ Avg Latency: {avg_latency_ms:.3f} ms per prompt")
print("-" * 45)
print("🧮 Confusion Matrix:")
print(f"   True Safe (Passed):      {conf_matrix[0][0]}")
print(f"   False Alarms (Blocked):  {conf_matrix[0][1]}")
print(f"   Missed Attacks (Passed): {conf_matrix[1][0]}")
print(f"   Caught Attacks (Blocked):{conf_matrix[1][1]}")
print("="*45 + "\n")