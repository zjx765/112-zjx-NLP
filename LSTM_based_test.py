import torch
import torch.nn as nn
import random
import os
import torch.nn.functional as F
import math
import sys
sys.path.append(os.path.abspath(".."))

def load_pretrained_matrix(path, expected_dim=300):
    """Helper function to safely load .pt file."""
    if path:
        print(f"Loading embeddings from {path}...")
        weights = torch.load(path)
        if weights.size(1) != expected_dim:
            raise ValueError(f"Mismatch: File is {weights.size(1)}d, expected {expected_dim}d")
        return weights
    return None

class word_embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, emb_path=None):
        super().__init__()
        
        # 1. Basic Embedding (300 dimensions)
        weights = load_pretrained_matrix(emb_path, emb_dim)
        if weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # 2. Projection Layer (adapter): 300 -> 512
        self.projection = nn.Linear(emb_dim, hidden_dim) if emb_dim != hidden_dim else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len]
        
        # [batch, seq_len, 300]
        x = self.embedding(x)
        
        # [batch, seq_len, 512]
        x = self.projection(x)
        
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, emb_path=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = word_embedding(
            vocab_size=input_dim,
            emb_dim=emb_dim,       # 300
            hidden_dim=hidden_dim, # 512
            dropout=dropout,
            emb_path=emb_path
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim, # Hidden size 512
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src shape: [batch_size, src_len]
        embedded = self.embedding(src)
        # outputs: All hidden states at each time step
        # hidden, cell: Final states (Context Vector)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, method="additive"):
        """Initialize the Attention mechanism"""
        super().__init__()
        self.method = method
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        if self.method == "dot":
            if self.enc_hid_dim != self.dec_hid_dim:
                raise ValueError("Dot-product attention requires enc_hid_dim == dec_hid_dim")
        elif self.method == "general":
            self.W = nn.Linear(self.enc_hid_dim, self.dec_hid_dim, bias=False)
        elif self.method == "additive":
            self.W_s = nn.Linear(self.dec_hid_dim, self.dec_hid_dim, bias=False)
            self.W_h = nn.Linear(self.enc_hid_dim, self.dec_hid_dim, bias=False)
            self.v = nn.Linear(self.dec_hid_dim, 1, bias=False)
        else:
            raise ValueError(f"Unknown attention method: {method}")

    def forward(self, hidden, encoder_outputs, mask=None):
        """Calculate attention weights"""
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        if self.method == "dot":
            query = hidden.unsqueeze(1)
            scale = math.sqrt(self.enc_hid_dim)
            energy = torch.bmm(query, encoder_outputs.permute(0, 2, 1)) / scale
        elif self.method == "general":
            query = hidden.unsqueeze(1)
            keys = self.W(encoder_outputs)
            energy = torch.bmm(query, keys.permute(0, 2, 1))
        elif self.method == "additive":
            query_proj = self.W_s(hidden).unsqueeze(1).repeat(1, src_len, 1)
            keys_proj = self.W_h(encoder_outputs)
            energy_tmp = torch.tanh(query_proj + keys_proj)
            energy = self.v(energy_tmp).permute(0, 2, 1)
        
        attention = F.softmax(energy.squeeze(1), dim=1)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            energy = energy.masked_fill(mask == 0, -1e10)
        
        return F.softmax(energy, dim=2).squeeze(1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention, emb_path=None):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.dec_hid_dim = dec_hid_dim
        self.embedding = word_embedding(
            vocab_size=output_dim,
            emb_dim=emb_dim,
            hidden_dim=dec_hid_dim, 
            dropout=dropout,
            emb_path=emb_path,
        )
        self.lstm = nn.LSTM(
            input_size=dec_hid_dim + enc_hid_dim,
            hidden_size=dec_hid_dim, 
            num_layers=n_layers, 
            dropout=dropout, 
            batch_first=True
        )
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs, mask=None):
        """Decoder forward pass with attention"""
        input = input.unsqueeze(1)
        embedded = self.embedding(input)
        a = self.attention(hidden[-1], encoder_outputs, mask)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted), dim=2).squeeze(1))
        return prediction, hidden, cell

class Seq2Seq_LSTM(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=0, sos_idx=2, eos_idx=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
    
    def create_mask(self, src):
        """Create a mask to ignore pad tokens"""
        return (src != self.pad_idx).to(self.device)
      
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """Training forward pass with teacher forcing"""
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        mask = self.create_mask(src)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        input = trg[:, 0] # <SOS>
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, mask)
            outputs[:, t, :] = output
            top1 = output.argmax(1) 
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else top1
            
        return outputs

    def predict_greedy(self, src_sentence, max_len=50):
        """Greedy decoding for prediction"""
        self.eval()
        src_tensor = torch.LongTensor(src_sentence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src_tensor)
            input_token = torch.tensor([self.sos_idx]).to(self.device)
            outputs = []
            for _ in range(max_len):
                output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
                prediction = output.argmax(1).item()
                if prediction == self.eos_idx:
                    break
                outputs.append(prediction)
                input_token = torch.tensor([prediction]).to(self.device)
        
        self.train()
        return outputs

    def predict_beam(self, src_sentence, beam_width=3, max_len=50):
        """Beam search for prediction"""
        self.eval()
        src_tensor = torch.LongTensor(src_sentence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src_tensor)
            beams = [(0.0, [self.sos_idx], (hidden, cell))]
            
            for _ in range(max_len):
                candidates = []
                for score, seq, (h, c) in beams:
                    if seq[-1] == self.eos_idx:
                        candidates.append((score, seq, (h, c)))
                        continue
                    
                    input_token = torch.tensor([seq[-1]]).to(self.device)
                    output, new_h, new_c = self.decoder(input_token, h, c, encoder_outputs)
                    log_probs = F.log_softmax(output, dim=1)
                    topk_probs, topk_ids = log_probs.topk(beam_width)
                    
                    for k in range(beam_width):
                        word_idx = topk_ids[0, k].item()
                        word_prob = topk_probs[0, k].item()
                        
                        new_score = score + word_prob
                        new_seq = seq + [word_idx]
                        candidates.append((new_score, new_seq, (new_h, new_c)))
                
                ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
                beams = ordered[:beam_width]
                
                if all([b[1][-1] == self.eos_idx for b in beams]):
                    break
            
            best_seq = beams[0][1]
        
        self.train()
        return best_seq[1:] # Remove <SOS>

