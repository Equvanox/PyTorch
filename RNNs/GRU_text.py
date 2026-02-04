import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------
# Simple tokenizer & vocab
# ---------------------------
sentences = [
    "i love deep learning",
    "deep learning loves data"
]

def tokenize(s):
    return s.split()

vocab = {"<pad>": 0, "<unk>": 1}
for s in sentences:
    for tok in tokenize(s):
        if tok not in vocab:
            vocab[tok] = len(vocab)

# Encode
encoded = [
    torch.tensor([vocab.get(t, vocab["<unk>"]) for t in tokenize(s)])
    for s in sentences
]

x = nn.utils.rnn.pad_sequence(encoded, batch_first=True)

print("Text tensor shape:", x.shape)

# ---------------------------
# Heatmap of input tokens
# ---------------------------
plt.imshow(x.float(), aspect='auto')
plt.title("Text Input Heatmap (Token IDs)")
plt.xlabel("Token Position")
plt.ylabel("Sentence Index")
plt.colorbar()
plt.show()

# ---------------------------
# GRU Model
# ---------------------------
class TextGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        return self.gru(x)

model = TextGRU(len(vocab), 16, 32)

output, h_n = model(x)

print("GRU output shape:", output.shape)

# ---------------------------
# Heatmap of GRU output
# ---------------------------
plt.imshow(output[0].detach().T, aspect='auto')
plt.title("GRU Output Heatmap (Hidden × Time)")
plt.xlabel("Token Position")
plt.ylabel("Hidden Units")
plt.colorbar()
plt.show()
