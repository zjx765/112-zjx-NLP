import os
import json
import gc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from gensim.models.fasttext import load_facebook_model
from tokenizers import Tokenizer
import torch.optim as optim
import time
import math
from LSTM_based_model import Attention, Encoder, Decoder, Seq2Seq_LSTM, word_embedding

# ================= 1. Global Configuration and Parameters =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Hyperparameters
BATCH_SIZE = 256
N_EPOCHS = 50           
LEARNING_RATE = 0.001
CLIP = 1.0
PATIENCE = 5            

# Model Dimensions
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HIDDEN_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5

# [Modification 1]: Corrected paths
# bin files are in the parent directory (..), but tokenizer is in current directory (.), and filenames have _L
EMB_TASKS = [
    ('../cc.zh.300.bin', 'tokenizer_zh_L.json', 'embeddings/embedding_zh.pt'),
    ('../cc.en.300.bin', 'tokenizer_en_L.json', 'embeddings/embedding_en.pt')
]

# ================= 2. Pre-processing Embeddings =================
def prepare_embeddings():
    os.makedirs('embeddings', exist_ok=True)
    for bin_path, tok_path, save_path in EMB_TASKS:
        if os.path.exists(save_path):
            print(f"✅ Detected {save_path} exists, skipping generation.")
            continue
        
        # Check if bin file exists
        if not os.path.exists(bin_path):
            print(f"⚠️ Warning: FastText vector file not found: {bin_path}, skipping Embedding generation.")
            continue

        print(f"🔨 Generating {save_path} (loading FastText, please wait)...")
        try:
            tokenizer = Tokenizer.from_file(tok_path)
            ft_model = load_facebook_model(bin_path)
            vocab = tokenizer.get_vocab()
            matrix = np.zeros((tokenizer.get_vocab_size(), ft_model.vector_size))
            for token, idx in vocab.items():
                word = token.replace('Ġ', '').replace('##', '').strip() or token
                try:
                    if token in ft_model.wv: matrix[idx] = ft_model.wv[token]
                    elif word in ft_model.wv: matrix[idx] = ft_model.wv[word]
                    else: matrix[idx] = ft_model.wv[word]
                except KeyError: pass
            torch.save(torch.FloatTensor(matrix), save_path)
            print(f"💾 Saved successfully: {save_path}")
            del ft_model; gc.collect()
        except Exception as e:
            print(f"❌ Failed to generate Embedding: {e}")

# ================= 3. Data Loading Pipeline =================
class TranslationDataset(Dataset):
    def __init__(self, path, max_len=128):
        self.data = []
        # Read line by line to avoid json.load Extra data error
        with open(path, 'r', encoding='utf-8') as f:
            print(f"🔍 Loading data file: {path} ...")
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    if 'zh_ids' in item and 'en_ids' in item:
                        if len(item['zh_ids']) <= max_len and len(item['en_ids']) <= max_len:
                            self.data.append({
                                'src': torch.LongTensor(item['zh_ids']),
                                'trg': torch.LongTensor(item['en_ids'])
                            })
                except Exception:
                    continue
        
        print(f"📦 Dataset loaded: {len(self.data)} valid samples")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]['src'], self.data[idx]['trg']

def get_dataloader(data_path, pad_idx, batch_size, shuffle=True):
    dataset = TranslationDataset(data_path)
    def collate_fn(batch):
        src, trg = zip(*batch)
        src_pad = pad_sequence(src, batch_first=True, padding_value=pad_idx)
        trg_pad = pad_sequence(trg, batch_first=True, padding_value=pad_idx)
        return src_pad, trg_pad
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# ================= 4. Initialize Environment =================
print("⚙️ Initializing environment configuration...")
try:
    # [Modification 2]: Corrected Tokenizer path (current directory, with _L suffix)
    tokenizer_zh = Tokenizer.from_file("tokenizer_zh_L.json")
    tokenizer_en = Tokenizer.from_file("tokenizer_en_L.json")
    
    INPUT_DIM = tokenizer_zh.get_vocab_size()
    OUTPUT_DIM = tokenizer_en.get_vocab_size()
    PAD_IDX = tokenizer_zh.token_to_id("<pad>")
    SOS_IDX = tokenizer_zh.token_to_id("<sos>")
    EOS_IDX = tokenizer_zh.token_to_id("<eos>")
    
    if PAD_IDX is None: PAD_IDX = 0
    if SOS_IDX is None: SOS_IDX = 2
    if EOS_IDX is None: EOS_IDX = 3
    print(f"✅ Tokenizer ready: PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}")

except Exception as e:
    print(f"⚠️ Tokenizer load failed ({e})")
    exit()

# ================= 5. Model Construction =================
def get_model(attn_method):
    path_zh = 'embeddings/embedding_zh.pt'
    path_en = 'embeddings/embedding_en.pt'
    # Add fault tolerance: if embedding generation failed, use random initialization
    path_zh = path_zh if os.path.exists(path_zh) else None
    path_en = path_en if os.path.exists(path_en) else None
    
    attn = Attention(HIDDEN_DIM, HIDDEN_DIM, method=attn_method)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, emb_path=path_zh)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, 
                  attention=attn, emb_path=path_en)
    model = Seq2Seq_LSTM(enc, dec, DEVICE, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX).to(DEVICE)
    return model

# ================= 6. Evaluation Function =================
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            output = model(src, trg, teacher_forcing_ratio=0) 
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)

# ================= 7. Training Engine =================
def run_training(exp_name, model, train_loader, valid_loader, n_epochs, tf_ratio):
    print(f"\n🚀 [Start Task] {exp_name} (Max Epochs={n_epochs}, Patience={PATIENCE})")
    print("-" * 60)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    save_dir = os.path.join('checkpoints', exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    best_valid_loss = float('inf')
    patience_counter = 0 
    
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = evaluate(model, valid_loader, criterion)
        
        mins = int((time.time() - start_time) / 60)
        valid_ppl = math.exp(min(avg_valid_loss, 100))
        
        print(f"Epoch {epoch+1:02} | Time: {mins}m | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_valid_loss:.3f} | Val PPL: {valid_ppl:.1f}")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"    🌟 Best Model Saved (Val Loss: {best_valid_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    ⚠️ No Improve. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping triggered! Training finished.")
            break

# ================= 8. Main Execution Logic =================
if __name__ == "__main__":
    prepare_embeddings()
    
    # [Modification 3]: Corrected data file paths (current directory, with _L suffix)
    train_file = 'train_data_tokenized_L.jsonl'
    valid_file = 'valid_tokenized_L.jsonl' 
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        print(f"❌ Data files not found: \nTrain: {train_file}\nValid: {valid_file}")
    else:
        print("📥 Loading DataLoaders...")
        train_loader = get_dataloader(train_file, PAD_IDX, BATCH_SIZE, shuffle=True)
        valid_loader = get_dataloader(valid_file, PAD_IDX, BATCH_SIZE, shuffle=False)
        
        EXPERIMENTS = [
            ('dot',      0.5, 'Exp1_Dot_Attn'),
            ('general',  0.5, 'Exp2_General_Attn'),
            ('additive', 0.5, 'Exp3_Additive_Attn'),
            ('additive', 0,   'Exp4_Additive_ZeroTF')
        ]
        
        print(f"🏭 Preparing to start {len(EXPERIMENTS)} experiments...")
        for attn_type, tf_val, name in EXPERIMENTS:
            print(f"🔄 Preparing experiment: {name} ...")
            if 'curr_model' in locals(): del curr_model
            gc.collect()
            torch.cuda.empty_cache()
            
            curr_model = get_model(attn_type)
            run_training(name, curr_model, train_loader, valid_loader, n_epochs=N_EPOCHS, tf_ratio=tf_val)
            
        print("\n🏆 All experiments finished！")