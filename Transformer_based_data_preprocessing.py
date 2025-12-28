import json
import os
import re
import unicodedata
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# ================= 1. Path Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

# Input Files: Directly point to the three independent files in Data directory
TRAIN_IN_FILE = os.path.join(DATA_DIR, "train_100k.jsonl")
VALID_IN_FILE = os.path.join(DATA_DIR, "valid.jsonl")
TEST_IN_FILE  = os.path.join(DATA_DIR, "test.jsonl")

# Output Files
TOKENIZER_ZH = os.path.join(CURRENT_DIR, 'tokenizer_zh_T.json')
TOKENIZER_EN = os.path.join(CURRENT_DIR, 'tokenizer_en_T.json')
TRAIN_OUT    = os.path.join(CURRENT_DIR, 'train_data_tokenized_T.jsonl')
VALID_OUT    = os.path.join(CURRENT_DIR, 'valid_tokenized_T.jsonl')
TEST_OUT     = os.path.join(CURRENT_DIR, 'test_tokenized_T.jsonl')

# Transformer specific special tokens
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

# ================= 2. Data Loading Functions =================
def clean_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_jsonl_dataset(file_path):
    """Generic loading function: read dataset by path"""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File {file_path} not found, skipping.")
        return []
    
    print(f"🔍 Reading: {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                en = item.get('english', item.get('en', '')).strip()
                zh = item.get('chinese', item.get('zh', '')).strip()
                if en and zh:
                    data.append({'en': clean_text(en), 'zh': clean_text(zh)})
            except Exception as e:
                print(f"   Line {i+1} format error, skipped: {e}")
    return data

# ================= 3. Train Tokenizer =================
def train_tokenizer(train_data, vocab_size=30000):
    """Train vocabulary using only training set"""
    print("🚀 Starting Tokenizer training (Transformer style)...")
    
    def train_single(sentences, filename):
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        tokenizer.decoder = decoders.Sequence([decoders.BPEDecoder(), decoders.Metaspace()])
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=SPECIAL_TOKENS, 
            min_frequency=2
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(filename)
        return tokenizer

    tok_en = train_single((x['en'] for x in train_data), TOKENIZER_EN)
    tok_zh = train_single((x['zh'] for x in train_data), TOKENIZER_ZH)
    return tok_en, tok_zh

# ================= 4. Process and Save =================
def process_and_save_subset(data, tok_en, tok_zh, out_path):
    """Tokenize input dataset and save"""
    if not data: return
    
    cls_en, sep_en = tok_en.token_to_id("[CLS]"), tok_en.token_to_id("[SEP]")
    cls_zh, sep_zh = tok_zh.token_to_id("[CLS]"), tok_zh.token_to_id("[SEP]")

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            en_ids = tok_en.encode(item['en']).ids
            zh_ids = tok_zh.encode(item['zh']).ids
            
            # Physically insert Transformer [CLS] and [SEP]
            out = {
                "en_ids": [cls_en] + en_ids + [sep_en],
                "zh_ids": [cls_zh] + zh_ids + [sep_zh],
                "en_raw": item['en'],
                "zh_raw": item['zh']
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    print(f"✅ Processing complete: {out_path} (Total {len(data)})")

if __name__ == "__main__":
    # 1. Load three files independently, no longer splitting from training set!
    train_data_raw = load_jsonl_dataset(TRAIN_IN_FILE)
    valid_data_raw = load_jsonl_dataset(VALID_IN_FILE)
    test_data_raw  = load_jsonl_dataset(TEST_IN_FILE)

    if train_data_raw:
        # 2. Train vocabulary
        tokenizer_en, tokenizer_zh = train_tokenizer(train_data_raw)
        
        # 3. Process and save to respective target files
        print("💾 Executing serialization save...")
        process_and_save_subset(train_data_raw, tokenizer_en, tokenizer_zh, TRAIN_OUT)
        process_and_save_subset(valid_data_raw, tokenizer_en, tokenizer_zh, VALID_OUT)
        process_and_save_subset(test_data_raw,  tokenizer_en, tokenizer_zh, TEST_OUT)
        
        print("\n✨ Transformer preprocessing flow successful! Used independent test/valid sets.")
    else:
        print("\n❌ Error: Training data not found, please check file paths in Data directory.")