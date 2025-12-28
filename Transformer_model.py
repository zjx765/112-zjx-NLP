import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Calculate RMS, keep dim for broadcasting
        # norm(x) = x / sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class NormChooser(nn.Module):
    def __init__(self, d_model, method='layer_norm'):
        super().__init__()
        if method == 'layer_norm':
            self.norm = nn.LayerNorm(d_model)
        elif method == 'rms_norm':
            self.norm = RMSNorm(d_model)
        else:
            raise ValueError("Method must be 'layer_norm' or 'rms_norm'")

    def forward(self, x):
        return self.norm(x)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000, mode='absolute', rel_max_dist=16):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.rel_max_dist = rel_max_dist

        # 1. Absolute Position Encoding (Sinusoidal)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('abs_pe', pe.unsqueeze(0))

        # 2. Relative Position Encoding (Learned Bias Table)
        # Relative distance range from -rel_max_dist to +rel_max_dist, total 2*rel_max_dist + 1 states
        self.num_rel_bins = 2 * rel_max_dist + 1
        self.rel_embedding = nn.Embedding(self.num_rel_bins, d_model // 8) # Relative dim is usually smaller

    def get_absolute_pe(self, x):
        """Return absolute position vector to add to Embedding"""
        return self.abs_pe[:, :x.size(1), :]

    def get_relative_bias(self, query_len, key_len):
        """
        Generate relative position bias matrix for subsequent Attention calculation
        Output shape: (query_len, key_len)
        """
        # Generate relative position matrix (i - j)
        range_q = torch.arange(query_len).unsqueeze(1)
        range_k = torch.arange(key_len).unsqueeze(0)
        rel_pos = range_q - range_k # shape (q, k)

        # Clamp to specified range and map to positive indices [0, num_rel_bins - 1]
        rel_pos = torch.clamp(rel_pos, -self.rel_max_dist, self.rel_max_dist)
        rel_pos_indices = rel_pos + self.rel_max_dist
        
        # Get relative position vector (Optional: return scalar bias directly)
        return rel_pos_indices.to(self.abs_pe.device)

    def forward(self, x):
        # If absolute position is included, add directly to input x
        if self.mode in ['absolute', 'both']:
            x = x + self.get_absolute_pe(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, pos_module = None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pos_module = pos_module

        # Define linear transformation matrices Q, K, V and final output projection W_o
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        batch_size = q.shape[0]
        
        # 1. Linear transformation and split into multiple heads
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Calculate scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Inject relative position bias (if mode includes relative)
        if self.pos_module is not None and self.pos_module.mode in ['relative', 'both']:
            q_len, k_len = Q.size(-2), K.size(-2)
            # Get relative position indices [q_len, k_len]
            rel_indices = self.pos_module.get_relative_bias(q_len, k_len)
            # Extract bias from rel_embedding in pos_module
            # Assume projecting it to a scalar bias added to score
            # Shape transform: [q_len, k_len, d_rel] -> [q_len, k_len]
            # Simplified here: directly map to head-related bias
            rel_bias = self.pos_module.rel_embedding(rel_indices) # [q_len, k_len, d_model//8]
            # We can turn it into (q_len, k_len) bias via simple linear layer or sum
            rel_scores = rel_bias.sum(dim=-1) 
            scores = scores + rel_scores # Broadcast to all batches and heads

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(scores, dim=-1)
        
        # 3. Multiply by V and concatenate heads
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc_out(out)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, pos_module, norm_method='layer_norm', dropout=0.1):
        """
        Args:
            pos_module: Pass in the previously defined UnifiedPositionEmbedding instance
            norm_method: 'layer_norm' or 'rms_norm'
        """
        super(EncoderLayer, self).__init__()
        
        # 1. Adapt MultiHeadAttention: receive pos_module to support relative position encoding
        self.attention = MultiHeadAttention(d_model, num_heads, pos_module)
        
        # 2. Adapt generic Norm: use NormChooser instead of hardcoded nn.LayerNorm
        self.norm1 = NormChooser(d_model, method=norm_method)
        self.norm2 = NormChooser(d_model, method=norm_method)
        
        # Feed Forward Network (Suggestion: modern NLP models often swap ReLU for GELU)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), 
            nn.Dropout(dropout), # Adding Dropout inside FFN is standard practice
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        Use Pre-Norm structure (x + Sublayer(Norm(x))) for better training stability
        """
        # --- Sublayer 1: Self-Attention ---
        # 1. Norm first
        norm_x = self.norm1(x)
        
        # 2. Attention Calc (Q, K, V all come from normed x)
        attn_out = self.attention(norm_x, norm_x, norm_x, mask)
        
        # 3. Residual connection (add to original x)
        x = x + self.dropout(attn_out)
        
        # --- Sublayer 2: Feed Forward ---
        # 1. Norm first
        norm_x = self.norm2(x)
        
        # 2. FFN Calc
        ffn_out = self.ffn(norm_x)
        
        # 3. Residual connection
        x = x + self.dropout(ffn_out)
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, pos_module, norm_method='layer_norm', dropout=0.1):
        """
        Args:
            pos_module: Relative position encoding module for Self-Attention
            norm_method: 'layer_norm' or 'rms_norm'
        """
        super(DecoderLayer, self).__init__()
        
        # 1. Adapt Self-Attention
        # Decoder self-attention requires Mask (prevent peeking) and can use relative position encoding
        self.self_attn = MultiHeadAttention(d_model, num_heads, pos_module=pos_module)
        
        # 2. Adapt Cross-Attention
        # Note: Cross-Attention usually doesn't use relative position encoding (pos_module=None),
        # Because the 'distance' definition between source (Key) and target (Query) is complex.
        # Here we mainly rely on absolute position encoding in input stage for alignment.
        self.cross_attn = MultiHeadAttention(d_model, num_heads, pos_module=None)
        
        # 3. Generic Norm Chooser (added a norm layer since Decoder has 3 sublayers)
        self.norm1 = NormChooser(d_model, method=norm_method)
        self.norm2 = NormChooser(d_model, method=norm_method)
        self.norm3 = NormChooser(d_model, method=norm_method)
        
        # 4. Feed Forward Network (Upgraded to GELU + internal Dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        Adopt Pre-Norm structure: x = x + Sublayer(Norm(x))
        """
        # --- Sublayer 1: Masked Self-Attention ---
        norm_x = self.norm1(x)
        # Note: Pass trg_mask here (Look-ahead Mask)
        self_attn_out = self.self_attn(norm_x, norm_x, norm_x, trg_mask)
        x = x + self.dropout(self_attn_out)
        
        # --- Sublayer 2: Cross-Attention ---
        norm_x = self.norm2(x)
        # Query from decoder (norm_x), Key/Value from encoder output (enc_out)
        # Pass src_mask here (mask Chinese Padding)
        cross_attn_out = self.cross_attn(norm_x, enc_out, enc_out, src_mask)
        x = x + self.dropout(cross_attn_out)
        
        # --- Sublayer 3: Feed Forward ---
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        
        return x


def make_src_mask(src, pad_idx):
    # src: [batch_size, src_len]
    # mask: [batch_size, 1, 1, src_len] (For broadcasting mechanism)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_trg_mask(trg, pad_idx):
    # trg: [batch_size, trg_len]
    N, trg_len = trg.shape
    
    # 1. Padding Mask
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2) # [N, 1, 1, len]
    
    # 2. Look-ahead Mask (Lower triangular matrix)
    trg_len = trg.shape[1]
    # torch.triu returns upper triangular, becomes lower after ==0 (keeps main diagonal and below)
    trg_sub_mask = torch.triu(torch.ones((trg_len, trg_len), device=trg.device), diagonal=1).bool()
    # Negate to get True for lower triangle (allowed positions)
    trg_sub_mask = ~trg_sub_mask # [len, len]
    
    # Combine both: must be non-padding and not seeing future
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask


class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx, 
        d_model=512, 
        num_layers=6, 
        num_heads=8, 
        d_ff=2048, 
        dropout=0.1, 
        max_len=100,
        norm_method='rms_norm',  # Optional 'layer_norm'
        pos_mode='absolute'      # Optional 'relative', 'both'
    ):
        super(Transformer, self).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.d_model = d_model
        
        # 1. Embedding Layer
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        
        # 2. Unified position encoding module (Shared or reused by Encoder and Decoder)
        self.pos_module = PositionEmbedding(d_model, max_len, mode=pos_mode)
        
        # 3. Stack Encoder and Decoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, self.pos_module, norm_method, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, self.pos_module, norm_method, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final Norm (Mandatory for Pre-Norm architecture)
        self.norm_chooser = NormChooser(d_model, norm_method) # Tool instance for creating norm layers
        self.encoder_final_norm = NormChooser(d_model, norm_method).norm # Directly get norm instance
        self.decoder_final_norm = NormChooser(d_model, norm_method).norm
        
        # 5. Output Projection
        self.fc_out = nn.Linear(d_model, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters (Xavier init is often important for Transformer)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        # src: [batch, src_len] -> [batch, src_len, d_model]
        # 1. Embedding + Absolute Pos (if enabled in pos_module)
        x = self.pos_module(self.src_embedding(src) * math.sqrt(self.d_model))
        x = self.dropout(x)
        
        # 2. Pass through Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
            
        # 3. Final Norm (Key to Pre-Norm architecture)
        return self.encoder_final_norm(x)

    def decode(self, trg, enc_out, src_mask, trg_mask):
        # trg: [batch, trg_len]
        x = self.pos_module(self.trg_embedding(trg) * math.sqrt(self.d_model))
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, trg_mask)
            
        # Final Norm
        return self.decoder_final_norm(x)

    def forward(self, src, trg):
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]
        """
        # 1. Generate Masks
        src_mask = make_src_mask(src, self.src_pad_idx)
        trg_mask = make_trg_mask(trg, self.trg_pad_idx)
        
        # 2. Encoder
        enc_out = self.encode(src, src_mask)
        
        # 3. Decoder
        dec_out = self.decode(trg, enc_out, src_mask, trg_mask)
        
        # 4. Output Projection
        output = self.fc_out(dec_out)
        return output