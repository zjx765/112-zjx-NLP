import os
import torch
import json
import re
import unicodedata
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu

# ================= Path Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "final_model") 
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Corresponds to the 'Data' folder in the screenshot (capitalized)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

# Directly use the standalone test set file
TEST_FILE = os.path.join(DATA_DIR, "test.jsonl") 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREFIX = "translate Chinese to English: "

def clean_text(text):
    """Keep cleaning logic consistent with training"""
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_test_data():
    if not os.path.exists(TEST_FILE):
         raise FileNotFoundError(f"❌ Test file not found: {TEST_FILE}")

    data = []
    print(f"📖 Loading test data from: {TEST_FILE}")
    
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                data.append(item)
            except:
                continue
    
    # Return all data directly here, as this is a dedicated test set
    return data 

def generate_translation(model, tokenizer, text):
    # Clean input and add prefix
    text = clean_text(text)
    text = PREFIX + text
    
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=4, 
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}, please run T5_train.py first")
        exit()

    print(f"🚀 Loading fine-tuned model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        exit()

    test_data = load_test_data()
    print(f"📦 Loaded {len(test_data)} test samples.")

    refs = []
    preds = []

    print("Running Inference...")
    
    # Iterate through the test set
    for item in tqdm(test_data): 
        src = item.get('chinese', item.get('zh', ''))
        tgt = item.get('english', item.get('en', ''))
        
        # Test only when both source and target sentences exist
        if src and tgt:
            pred = generate_translation(model, tokenizer, src)
            
            preds.append(pred)
            # Clean reference answers to ensure fair BLEU calculation (remove extra spaces)
            refs.append(clean_text(tgt))

    if refs:
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        print("\n" + "="*40)
        print(f"🏆 T5-Finetuned Evaluation Results")
        print("="*40)
        print(f"✅ BLEU Score: {bleu.score:.2f}")
        
        print("\n📝 Sample Comparisons (Top 5):")
        for i in range(min(5, len(preds))):
            print(f"Input:  {clean_text(test_data[i].get('chinese', test_data[i].get('zh')))}")
            print(f"Ref:    {refs[i]}")
            print(f"T5 Out: {preds[i]}")
            print("-" * 30)