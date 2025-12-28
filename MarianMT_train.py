import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python' 
import json
import torch
import re
import unicodedata
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

# ================= Path Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Corresponds to the 'Data' folder in the screenshot (capitalized)
DATA_DIR = os.path.join(ROOT_DIR, 'Data') 

# Explicitly specify training and validation files
TRAIN_FILE = os.path.join(DATA_DIR, "train_100k.jsonl")
VALID_FILE = os.path.join(DATA_DIR, "valid.jsonl")

# ================= Configuration =================
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-zh-en"
BATCH_SIZE = 16 
MAX_LENGTH = 128
EPOCHS = 20
LEARNING_RATE = 5e-5
PREFIX = ""

# ================= Text Cleaning =================
def clean_text(text):
    """
    Basic Cleaning:
    1. Unicode normalization (NFKC) to fix full-width characters etc.
    2. Remove extra whitespace (newlines, tabs converted to single spaces)
    3. Strip leading and trailing spaces
    """
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data_from_file(filepath):
    """Read and clean data from the specified file"""
    print(f"📖 Loading data from: {filepath}")
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        # Read jsonl line by line
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # Handle different key names
                en = item.get('english', item.get('en', ''))
                zh = item.get('chinese', item.get('zh', ''))
                
                # Clean data
                en_clean = clean_text(en)
                zh_clean = clean_text(zh)
                
                if en_clean and zh_clean:
                    # Model input requires prefix (if any)
                    data.append({"zh": PREFIX + zh_clean, "en": en_clean})
            except json.JSONDecodeError:
                print(f"⚠️ Skipping error line {i+1}")
                continue
    
    print(f"✅ Loaded {len(data)} valid pairs from {os.path.basename(filepath)}")
    return data

def preprocess_function(examples):
    # Input processing
    inputs = examples["zh"]
    targets = examples["en"]
    
    # Tokenization (automatically handle padding and truncation)
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == "__main__":
    print("🚀 Downloading/Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 1. Load independent datasets
    train_raw = load_data_from_file(TRAIN_FILE)
    val_raw = load_data_from_file(VALID_FILE)

    print(f"📊 Final Dataset Size -> Train: {len(train_raw)}, Valid: {len(val_raw)}")

    # 2. Convert to HuggingFace Dataset
    train_ds = Dataset.from_list(train_raw)
    val_ds = Dataset.from_list(val_raw)

    # 3. Preprocessing (Tokenization)
    print("⚙️ Tokenizing data...")
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)

    # 4. Training Arguments
    args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(CURRENT_DIR, "t5_checkpoints"),
        evaluation_strategy="epoch",  # Note version compatibility, changed previously
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        
        fp16=False,  # <--- ❌ Force disable fp16 to fix NaN issue
        
        logging_dir=os.path.join(CURRENT_DIR, "logs"),
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("🔥 Starting Fine-tuning...")
    trainer.train()

    # 5. Save Model
    final_path = os.path.join(CURRENT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"✅ Model saved to {final_path}")