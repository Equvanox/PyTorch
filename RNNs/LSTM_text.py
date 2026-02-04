import torch
import torch.nn as nn

# ---------------------------
# 1. Sample text
# ---------------------------
sentences = [
    "i love deep learning",
    "deep learning loves data"
]

# ---------------------------
# 2. Tokenizer
# ---------------------------
def tokenize(text):
    return text.lower().split()

# ---------------------------
# 3. Build vocabulary (pure Python)
# ---------------------------
def build_vocab(sentences, specials=["<pad>", "<unk>"]):
    vocab = {}
    idx = 0

    for token in specials:
        vocab[token] = idx
        idx += 1

    for sentence in sentences:
        for token in tokenize(sentence):
            if token not in vocab:
                vocab[token] = idx
                idx += 1

    return vocab

vocab = build_vocab(sentences)
PAD_IDX = vocab["<pad>"]

print("Vocabulary:", vocab)

# ---------------------------
# 4. Encode text
# ---------------------------
encoded = [
    torch.tensor([vocab.get(tok, vocab["<unk>"]) for tok in tokenize(s)])
    for s in sentences
]

# Pad sequences
x = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=PAD_IDX)

print("Input tensor shape:", x.shape)
print(x)

# ---------------------------
# 5. LSTM Model
# ---------------------------
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        return self.lstm(x)

model = TextLSTM(
    vocab_size=len(vocab),
    embed_dim=16,
    hidden_size=32
)

# ---------------------------
# 6. Forward pass
# ---------------------------
output, (h_n, c_n) = model(x)

print("Output shape:", output.shape)     # [batch, seq_len, hidden]
print("Hidden shape:", h_n.shape)        # [num_layers, batch, hidden]
print("Cell shape:", c_n.shape)
