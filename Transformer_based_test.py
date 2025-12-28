import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import json
import os
import sys
import re
import pandas as pd
from tqdm import tqdm

# ================= 0. Environment Setup =================
try:
    import sacrebleu
except ImportError:
    print("⚠️ Warning: sacrebleu not found. BLEU score calculation will be skipped.")
    sacrebleu = None

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from Transformer_model import Transformer, make_src_mask, make_trg_mask
except ImportError:
    raise ImportError("❌ Cannot import Transformer_model, please ensure Transformer_model.py is in the same directory.")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 1. Helper Functions =================
def manual_detokenize(text):
    text = text.replace(" ##", "").replace("##", "")
    text = text.replace("Ġ", "") 
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    for i in range(max_len - 1):
        tgt_mask = make_trg_mask(ys, 0)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        prob = model.fc_out(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol: break
    return ys

# ================= 2. Core Evaluator Class =================
class Evaluator:
    def __init__(self):
        self.zh_path = os.path.join(current_dir, "tokenizer_zh_T.json")
        self.en_path = os.path.join(current_dir, "tokenizer_en_T.json")
        self.test_file = os.path.join(current_dir, "test_tokenized_T.jsonl")

        if os.path.exists(self.zh_path) and os.path.exists(self.en_path):
            self.tok_zh = Tokenizer.from_file(self.zh_path)
            self.tok_en = Tokenizer.from_file(self.en_path)
        else:
            raise FileNotFoundError("❌ Tokenizers not found.")

        self.sos_idx = self.tok_en.token_to_id("[CLS]") 
        self.eos_idx = self.tok_en.token_to_id("[SEP]") 
        self.pad_idx = self.tok_en.token_to_id("[PAD]") 
        if self.sos_idx is None: self.sos_idx = 1
        if self.eos_idx is None: self.eos_idx = 2
        if self.pad_idx is None: self.pad_idx = 0

        self.test_samples = []
        if os.path.exists(self.test_file):
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip(): self.test_samples.append(json.loads(line))
            print(f"📦 Loaded {len(self.test_samples)} test samples.")
        else:
            print(f"⚠️ Test file not found: {self.test_file}")

    def evaluate_experiment(self, exp_name, config):
        print(f"\n🚀 Evaluating Experiment: {exp_name}")
        
        try:
            # Dynamic instantiation, ensuring parameters match
            model = Transformer(
                src_vocab_size=self.tok_zh.get_vocab_size(),
                trg_vocab_size=self.tok_en.get_vocab_size(),
                src_pad_idx=self.pad_idx,
                trg_pad_idx=self.pad_idx,
                d_model=config['d_model'],
                num_layers=config['n_layers'],
                num_heads=config['n_heads'],
                d_ff=config['d_model'] * 4, # Keep * 4
                dropout=0,
                max_len=config.get('max_len', 200), # Dynamic read
                norm_method=config['norm_method'], 
                pos_mode=config['pos_mode']
            ).to(DEVICE)
        except Exception as e:
            print(f"   ❌ Model init failed: {e}")
            return None

        ckpt_path = os.path.join(current_dir, "checkpoints", exp_name, "best_model.pt")
        if os.path.exists(ckpt_path):
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                model.eval()
            except Exception as e:
                print(f"   ❌ Weight loading failed: {e}")
                return None
        else:
            print(f"   ⚠️ Checkpoint not found: {ckpt_path}")
            return None

        preds, refs = [], []
        # Use all test data by default
        samples_to_test = self.test_samples

        for item in tqdm(samples_to_test, desc=f"   -> Inferencing", leave=False):
            zh_ids = item['zh_ids']
            en_ids = item['en_ids']
            
            src_tensor = torch.tensor(zh_ids).unsqueeze(0).to(DEVICE)
            src_mask = make_src_mask(src_tensor, self.pad_idx)
            
            with torch.no_grad():
                out_tokens = greedy_decode(
                    model, src_tensor, src_mask, max_len=100, 
                    start_symbol=self.sos_idx, end_symbol=self.eos_idx
                )
            
            out_ids = out_tokens.squeeze().tolist()
            if isinstance(out_ids, int): out_ids = [out_ids]
            
            filtered_ids = []
            for idx in out_ids:
                if idx == self.sos_idx: continue
                if idx == self.eos_idx: break
                filtered_ids.append(idx)
            
            pred_text = self.tok_en.decode(filtered_ids)
            ref_text = self.tok_en.decode(en_ids)
            preds.append(manual_detokenize(pred_text))
            refs.append(manual_detokenize(ref_text))

        score = 0.0
        if sacrebleu and refs:
            score = sacrebleu.corpus_bleu(preds, [refs], tokenize='13a').score
        
        print(f"   ✅ BLEU Score: {score:.2f}")
        
        return {
            "Experiment": exp_name,
            "BLEU": round(score, 2),
            "Config": str(config),
            "Sample Pred": preds[0] if preds else ""
        }

if __name__ == "__main__":
    # Config must match train.py
    BASE_CONFIG = {
        'd_model': 512, 'n_layers': 6, 'n_heads': 8, 'dropout': 0.1,
        'norm_method': 'layer_norm', 'pos_mode': 'absolute',
        'batch_size': 64, 'lr': 0.0005, 'epochs': 30, 'max_len': 200
    }

    experiments_list = [
        ("baseline", {}), 
        ("pos_relative", {'pos_mode': 'relative'}),
        ("norm_rms", {'norm_method': 'rms_norm'}),
        ("bs_32", {'batch_size': 32}),
        ("bs_128", {'batch_size': 128}), 
        ("lr_1e-3", {'lr': 0.001}),
        ("lr_1e-4", {'lr': 0.0001}),
        ("model_small", {'d_model': 256, 'n_layers': 3, 'n_heads': 4}),
    ]

    evaluator = Evaluator()
    results = []

    print(f"📊 Starting ablation study evaluation ({len(experiments_list)} groups)...")
    for exp_name, override_params in experiments_list:
        current_config = BASE_CONFIG.copy()
        current_config.update(override_params)
        res = evaluator.evaluate_experiment(exp_name, current_config)
        if res: results.append(res)
    
    print("\n" + "="*60)
    print("🏆 Final Ablation Study Report")
    if results:
        df = pd.DataFrame(results)
        cols = ["Experiment", "BLEU", "Sample Pred"]
        print(df[cols].to_markdown(index=False))
        df.to_csv(os.path.join(current_dir, "ablation_results.csv"), index=False)
    else:
        print("❌ No valid results generated")