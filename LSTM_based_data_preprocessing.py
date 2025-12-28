import json
import os
import re
import unicodedata
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')

TRAIN_IN = os.path.join(DATA_DIR, 'train_100k.jsonl')
VALID_IN = os.path.join(DATA_DIR, 'valid.jsonl')
TEST_IN  = os.path.join(DATA_DIR, 'test.jsonl')

TOKENIZER_ZH = os.path.join(CURRENT_DIR, 'tokenizer_zh_L.json')
TOKENIZER_EN = os.path.join(CURRENT_DIR, 'tokenizer_en_L.json')
TRAIN_OUT    = os.path.join(CURRENT_DIR, 'train_data_tokenized_L.jsonl')
VALID_OUT    = os.path.join(CURRENT_DIR, 'valid_tokenized_L.jsonl')
TEST_OUT     = os.path.join(CURRENT_DIR, 'test_tokenized_L.jsonl')

SPECIAL_TOKENS = ["<pad>", "<unk>", "<sos>", "<eos>"]

def clean_text(text):
    """Normalize text by removing extra spaces."""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File not found {file_path}")
        return []
    
    data = []
    print(f"🔍 Loading: {file_path}")
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
                print(f"   Skipping line {i+1}: {e}")
    return data

# ================= 3. Tokenizer Training =================
def train_tokenizer(train_data, vocab_size=30000):
    """Train the tokenizer using only the training data."""
    print("🚀 Starting tokenizer training...")
    
    def train_single(sentences, filename):
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=SPECIAL_TOKENS, 
            min_frequency=2,
            end_of_word_suffix="</w>"
        )
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(filename)
        return tokenizer

    tok_en = train_single((x['en'] for x in train_data), TOKENIZER_EN)
    tok_zh = train_single((x['zh'] for x in train_data), TOKENIZER_ZH)
    return tok_en, tok_zh

# ================= 4. Processing and Saving =================
def process_and_save_subset(data, tok_en, tok_zh, out_path):
    """Encode the data and save it to a file."""
    if not data: return
    
    sos_en, eos_en = tok_en.token_to_id("<sos>"), tok_en.token_to_id("<eos>")
    sos_zh, eos_zh = tok_zh.token_to_id("<sos>"), tok_zh.token_to_id("<eos>")

    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            en_ids = tok_en.encode(item['en']).ids
            zh_ids = tok_zh.encode(item['zh']).ids
            
            out = {
                "en_ids": [sos_en] + en_ids + [eos_en],
                "zh_ids": [sos_zh] + zh_ids + [eos_zh],
                "en_raw": item['en'],
                "zh_raw": item['zh']
            }
            f.write(json.dumps(out, ensure_ascii=False) + '\n')
    print(f"✅ Saved to: {out_path} (Count: {len(data)})")

if __name__ == "__main__":
    # 1. Load the three separate files
    train_data = load_jsonl(TRAIN_IN)
    valid_data = load_jsonl(VALID_IN)
    test_data  = load_jsonl(TEST_IN)

    if train_data:
        # 2. Train the vocabulary using the training set
        t_en, t_zh = train_tokenizer(train_data)
        
        # 3. Process the three datasets and save them
        process_and_save_subset(train_data, t_en, t_zh, TRAIN_OUT)
        process_and_save_subset(valid_data, t_en, t_zh, VALID_OUT)
        process_and_save_subset(test_data,  t_en, t_zh, TEST_OUT)
        
        print("\n✨ Preprocessing complete!")
