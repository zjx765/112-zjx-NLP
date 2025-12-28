# 112-zjx-NLP
# Files architecture:
├── Data/                       # Dataset directory
│   ├── test.jsonl              # Test set (Chinese-English pairs)
│   ├── train_10k.jsonl         # Small training set for debugging
│   ├── train_100k.jsonl        # Main training set
│   └── valid.jsonl             # Validation set
│
├── LSTM-based/                 # 1. Seq2Seq LSTM with Attention
│   ├── checkpoints/            # Saved model weights
│   ├── embeddings/             # Pre-trained FastText embeddings
│   ├── LSTM-result/            # Training/Testing logs
│   ├── LSTM_based_model.py     # Model definition (Encoder, Decoder, Attention)
│   ├── LSTM_based_train.py     # Training loop implementation
│   ├── LSTM_based_test.py      # Inference and BLEU evaluation
│   └── *_preprocessing.py      # Data processing & Embedding generation
│
├── Transformer-based/          # 2. Custom Transformer Implementation\
|   ├── checkpoints/            # Saved model weights
|   ├── Transformer-result/     # Results storage
|   ├── ablation_results.csv    # CSV report for ablation studies
|   ├── Transformer_model.py    # Transformer architecture (Self-Attention, PE, etc.)
|   ├── Transformer_based_train.py # Training loop with Early Stopping
|   ├── Transformer_based_test.py  # Inference, greedy decode, and ablation eval
|   └── *_preprocessing.py      # Tokenizer training (BPE) & serialization
│
└── MarianMT-pretraining/       # 3. Fine-tuning MarianMT (Hugging Face)
    ├── final_model/            # Saved fine-tuned model & tokenizer
    ├── MarianMT_checkpoints/   # Intermediate training checkpoints
    ├── MarianMT_result/        # Output logs
    ├── MarianMT_train.py       # Fine-tuning script using HF Trainer
    └── MarianMT_test.py        # Inference script using HF Pipeline

# Notice
# Here is just the code file of all the models of the NLP-project, because the weight file is too large, it is not possible to upload them all, the specific model weights I save in Baidu Netdisk, thank you for your understanding.
通过网盘分享的文件：NLP-proj-files-250010112-ZJX
链接: https://pan.baidu.com/s/1jG4FfJ2uu5c3CWlIxwhAZQ?pwd=kvws 提取码: kvws 
--来自百度网盘超级会员v6的分享

# Module Descriptions
1. Data Processing (Data/)
Centralized data storage ensuring all models are trained and evaluated on identical splits.
Format: Standardized JSONL (JSON Lines) format to handle variable-length text efficiently.
Splits: Includes a massive training set (train_100k.jsonl), a rapid-prototyping subset (train_10k.jsonl), and dedicated validation/test sets (valid.jsonl, test.jsonl) to monitor overfitting and generalization.

2. RNN-Based Architecture (LSTM-based/)
A classic Sequence-to-Sequence (Seq2Seq) implementation designed to benchmark recurrent neural networks against modern attention mechanisms.
Encoder-Decoder: Utilizes a Bi-directional LSTM encoder to capture sentence context from both directions and a uni-directional LSTM decoder.
Attention Mechanisms: Features custom implementations of multiple attention scoring methods (Dot-Product, General, and Additive/Bahdanau) to solve the bottleneck problem of fixed-length context vectors.
Embeddings: Integrates pre-trained FastText word vectors to handle semantic similarities and out-of-vocabulary (OOV) tokens effectively.

3.  Custom Transformer (Transformer-based/)
A fully custom, "from-scratch" implementation of the Transformer architecture (Vaswani et al., 2017), designed for deep interpretability and experimentation.
Core Architecture: Implements Multi-Head Self-Attention (MHA), Position-wise Feed-Forward Networks, and Residual Connections.
Advanced Features:
Tokenizer: Includes a custom-trained Byte-Pair Encoding (BPE) tokenizer (tokenizers library) tailored specifically for this dataset.
Configurable Normalization: Supports switching between LayerNorm and RMSNorm for training stability analysis.
Position Encodings: Implements both standard Sinusoidal (Absolute) and Relative position encodings.
Ablation Studies: Contains a specialized testing suite (Transformer_based_test.py) to rigorously evaluate the impact of hyperparameters (e.g., batch size, learning rate, depth vs. width) on BLEU scores.

4.  Transfer Learning Baseline (MarianMT-pretraining/)
Implements a State-of-the-Art (SOTA) baseline using the Hugging Face Transformers ecosystem.
Model: Fine-tunes the MarianMT (Helsinki-NLP/opus-mt-zh-en) architecture, a highly efficient Transformer originally trained on the Opus corpus.
Pipeline: leverages Seq2SeqTrainer for optimized training loops, mixed-precision training (FP16), and standardized evaluation metrics. This module serves as the performance "ceiling" to evaluate the effectiveness of the custom implementations.

