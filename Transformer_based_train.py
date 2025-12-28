import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import json
import os
import sys
import time
from tqdm import tqdm
from Transformer_model import Transformer  # Ensure Transformer_model.py is in the same directory

# ==========================================
# 1. Basic Utility Classes
# ==========================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)

class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ File not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        self.data.append(json.loads(line))
                    except: continue

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['zh_ids'], dtype=torch.long), \
               torch.tensor(self.data[idx]['en_ids'], dtype=torch.long)

def get_collate_fn(pad_idx):
    def collate_fn(batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(src_sample)
            trg_batch.append(trg_sample)
        src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
        trg_batch = pad_sequence(trg_batch, padding_value=pad_idx, batch_first=True)
        return src_batch, trg_batch
    return collate_fn

def load_tokenizer_info(zh_path, en_path):
    if not os.path.exists(zh_path) or not os.path.exists(en_path):
        print(f"❌ Error: Tokenizers not found.")
        sys.exit(1)
    tok_zh = Tokenizer.from_file(zh_path)
    tok_en = Tokenizer.from_file(en_path)
    pad_id_zh = tok_zh.token_to_id("[PAD]") if tok_zh.token_to_id("[PAD]") is not None else 0
    pad_id_en = tok_en.token_to_id("[PAD]") if tok_en.token_to_id("[PAD]") is not None else 0
    return {
        "src_vocab_size": tok_zh.get_vocab_size(),
        "trg_vocab_size": tok_en.get_vocab_size(),
        "src_pad_idx": pad_id_zh,
        "trg_pad_idx": pad_id_en
    }

# ==========================================
# 2. Training and Evaluation Logic
# ==========================================

def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    # Disabled tqdm detailed bar to keep logs clean
    for src, trg in tqdm(dataloader, desc="Train", leave=False, disable=True):
        src, trg = src.to(device), trg.to(device)
        trg_input = trg[:, :-1]
        trg_label = trg[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, trg_input)
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_label = trg_label.contiguous().view(-1)
        
        loss = criterion(output, trg_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_label = trg[:, 1:]
            output = model(src, trg_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_label = trg_label.contiguous().view(-1)
            loss = criterion(output, trg_label)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# ==========================================
# 3. Single Experiment Runner
# ==========================================

def run_experiment(exp_name, config):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"🧪 Experiment: {exp_name}")
    print(f"⚙️ Config: {json.dumps(config, indent=2, ensure_ascii=False)}")
    print(f"{'='*80}")

    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "train_data_tokenized_T.jsonl")
    valid_path = os.path.join(current_dir, "valid_tokenized_T.jsonl")
    tok_zh_path = os.path.join(current_dir, "tokenizer_zh_T.json")
    tok_en_path = os.path.join(current_dir, "tokenizer_en_T.json")

    # 2. Data
    meta = load_tokenizer_info(tok_zh_path, tok_en_path)
    collate_fn = get_collate_fn(meta['trg_pad_idx'])

    train_ds = TokenizedDataset(train_path)
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    if os.path.exists(valid_path):
        valid_ds = TokenizedDataset(valid_path)
        valid_dl = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
    else:
        valid_dl = None
        print("⚠️ No validation set found.")

    # 3. Model
    model = Transformer(
        src_vocab_size=meta['src_vocab_size'],
        trg_vocab_size=meta['trg_vocab_size'],
        src_pad_idx=meta['src_pad_idx'],
        trg_pad_idx=meta['trg_pad_idx'],
        d_model=config['d_model'],
        num_layers=config['n_layers'],
        num_heads=config['n_heads'],
        # [Critical] Keep * 4
        d_ff=config['d_model'] * 4,
        dropout=config['dropout'],
        # [Critical] Read max_len from config, default 200
        max_len=config.get('max_len', 200),
        norm_method=config['norm_method'], 
        pos_mode=config['pos_mode']
    ).to(DEVICE)

    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)

    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=meta['trg_pad_idx'], label_smoothing=0.1)

    # 5. Save
    save_dir = os.path.join(current_dir, "checkpoints", exp_name)
    best_model_path = os.path.join(save_dir, "best_model.pt")
    early_stopping = EarlyStopping(patience=5, save_path=best_model_path)

    # 6. Loop
    start_time = time.time()
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, 1.0, DEVICE)
        
        if valid_dl:
            val_loss = evaluate(model, valid_dl, criterion, DEVICE)
            print(f"[{exp_name}] Epoch {epoch+1:02} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"[{exp_name}] Epoch {epoch+1:02} | Train Loss: {train_loss:.4f}")
            torch.save(model.state_dict(), best_model_path)

    total_mins = int((time.time() - start_time) / 60)
    print(f"✅ Finished {exp_name} in {total_mins} mins.")


if __name__ == "__main__":
    # --- Ablation Study Config ---
    BASE_CONFIG = {
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 4,
        'dropout': 0.3,
        'norm_method': 'layer_norm', 
        'pos_mode': 'absolute', 
        'batch_size': 128, 
        'lr': 0.0003, 
        'epochs': 50,
        'max_len': 150  # Default length
    }

    # Experiments List
    experiments = [
        ("baseline", {}), 
        ("pos_relative", {'pos_mode': 'relative'}),
        ("norm_rms", {'norm_method': 'rms_norm'}),
        # Note VRAM: Comment out this line if 128 causes OOM
        ("bs_32", {'batch_size': 32}),
        ("bs_128", {'batch_size': 128}), 
        ("lr_1e-3", {'lr': 0.001}),
        ("lr_1e-4", {'lr': 0.0001}),
        ("model_small", {'d_model': 256, 'n_layers': 3, 'n_heads': 4}),
    ]

    print(f"🚀 Preparing to start {len(experiments)} ablation experiments...")
    for exp_name, override_params in experiments:
        current_config = BASE_CONFIG.copy()
        current_config.update(override_params)
        try:
            run_experiment(exp_name, current_config)
        except Exception as e:
            print(f"❌ Experiment {exp_name} failed: {e}")
            continue

    print("\n🏆 All ablation experiments finished!")