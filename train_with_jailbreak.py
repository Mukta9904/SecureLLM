import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset

MODEL_DIR = "new_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ 1. Assembling the Ultimate Datasets...")

# --- LISTS TO HOLD OUR DATA ---
train_texts, train_labels = [], []
test_texts, test_labels = [], []

# Helper function to parse xTRam1's specific format
def parse_xtram1(ds_split):
    X, y = [], []
    cols = ds_split.column_names
    text_col = 'prompt' if 'prompt' in cols else 'text' if 'text' in cols else cols[0]
    label_col = 'label' if 'label' in cols else 'is_injection' if 'is_injection' in cols else cols[1]
    
    for row in ds_split:
        text = row[text_col]
        label = row[label_col]
        if not isinstance(text, str) or not text.strip(): continue
        
        is_attack = 1
        if isinstance(label, str) and label.lower() in ['safe', '0', 'false']: is_attack = 0
        elif isinstance(label, (int, float, bool)) and (label == 0 or label is False): is_attack = 0
        
        X.append(text)
        y.append(is_attack)
    return X, y

# A. xTRam1 Dataset (Train & Test)
try:
    print("   -> Fetching xTRam1...")
    xt_tr, yl_tr = parse_xtram1(load_dataset("xTRam1/safe-guard-prompt-injection", split="train"))
    train_texts.extend(xt_tr); train_labels.extend(yl_tr)
    
    xt_te, yl_te = parse_xtram1(load_dataset("xTRam1/safe-guard-prompt-injection", split="test"))
    test_texts.extend(xt_te); test_labels.extend(yl_te)
except Exception as e: print(f"   ⚠️ xTRam1 Error: {e}")

# B. Deepset Dataset (Train & Test)
try:
    print("   -> Fetching Deepset...")
    ds_deepset_train = load_dataset("deepset/prompt-injections", split="train")
    train_texts.extend(list(ds_deepset_train['text']))
    train_labels.extend([1] * len(ds_deepset_train)) # 1 = Attack
    
    ds_deepset_test = load_dataset("deepset/prompt-injections", split="test")
    test_texts.extend(list(ds_deepset_test['text']))
    test_labels.extend([1] * len(ds_deepset_test))
except Exception as e: print(f"   ⚠️ Deepset Error: {e}")

# C. Jailbreaks Dataset (Train only - they don't have a test split)
try:
    print("   -> Fetching Jailbreaks...")
    ds_jailbreaks = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
    col_name = 'Prompt' if 'Prompt' in ds_jailbreaks.column_names else ds_jailbreaks.column_names[0]
    jailbreaks = list(ds_jailbreaks[col_name])
    train_texts.extend(jailbreaks)
    train_labels.extend([1] * len(jailbreaks))
except Exception as e: print(f"   ⚠️ Jailbreak Error: {e}")

# D. Balance with Alpaca (Safe Prompts)
print("⚖️ Balancing the datasets with Alpaca...")
try:
    # Calculate how many safe prompts we are missing to make it 50/50
    needed_train_safe = max(0, train_labels.count(1) - train_labels.count(0))
    needed_test_safe = max(0, test_labels.count(1) - test_labels.count(0))
    total_needed = needed_train_safe + needed_test_safe
    
    if total_needed > 0:
        ds_alpaca = load_dataset("tatsu-lab/alpaca", split=f"train[:{total_needed}]")
        alpaca_texts = [row['instruction'] + " " + row['input'] for row in ds_alpaca]
        
        train_texts.extend(alpaca_texts[:needed_train_safe])
        train_labels.extend([0] * needed_train_safe)
        
        test_texts.extend(alpaca_texts[needed_train_safe:])
        test_labels.extend([0] * needed_test_safe)
except Exception as e: print(f"   ⚠️ Alpaca Error: {e}")

# E. THE SMALL TALK INJECTION (FIX FOR "HELLO")
print("🗣️ Injecting safe conversational data...")
small_talk = [
    "hello", "hi", "hey", "hello there", "good morning", "good evening", 
    "how are you", "thanks", "thank you", "testing", "just testing", 
    "what's up", "can you help me", "help", "hi aegis"
] * 200 # Multiply by 200 to give it enough weight in the math

train_texts.extend(small_talk)
train_labels.extend([0] * len(small_talk)) # 0 = Safe


print(f"\n📊 FINAL TRAIN DATA: {len(train_texts)} prompts...")
print(f"\n📊 FINAL TRAIN DATA: {len(train_texts)} prompts ({train_labels.count(1)} Attacks, {train_labels.count(0)} Safe)")
print(f"📊 FINAL TEST DATA:  {len(test_texts)} prompts ({test_labels.count(1)} Attacks, {test_labels.count(0)} Safe)")

# --- TRAINING ---
print("\n🧠 2. Training Ultimate Model...")
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_features=12000)
X_train_vectors = vectorizer.fit_transform(train_texts)

classifier = LogisticRegression(C=1.0, max_iter=1000)
classifier.fit(X_train_vectors, train_labels)

# --- SAVING ---
print("\n💾 3. Saving Models and Test Vault...")
with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f: pickle.dump(vectorizer, f)
with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f: pickle.dump(classifier, f)

# Lock the specific test splits in the vault
with open(os.path.join(MODEL_DIR, "test_data.pkl"), "wb") as f:
    pickle.dump({"X_test": test_texts, "y_test": test_labels}, f)

print("🎉 DONE! Ultimate Vault created.")