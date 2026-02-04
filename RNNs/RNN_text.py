import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt


# 1. Setup: Let's assume a tiny vocabulary
# "The" = 0, "cat" = 1, "sat" = 2
sentence = torch.tensor([0, 1, 2]) 

# 2. Define Layers
vocab_size = 3
embed_dim = 5  # Each word represented by 5 numbers
hidden_dim = 10 # The size of the "memory"

embedding_layer = nn.Embedding(vocab_size, embed_dim)
rnn_cell = nn.RNN(embed_dim, hidden_dim, batch_first=True)

# 3. Process the sentence
# Convert words to vectors
inputs = embedding_layer(sentence).unsqueeze(0) # Shape: [1, 3, 5] (Batch, Sequence, Features)

# Initialize memory (hidden state) as zeros
h0 = torch.zeros(1, 1, hidden_dim)

# Pass through RNN
# 'out' contains the hidden state after EACH word
# 'hn' contains the final hidden state (the "final memory")
out, hn = rnn_cell(inputs, h0)

print(f"Sentence length: {out.shape[1]}")
print(f"Memory after 'The':\n{out[0, 0, :]}")
print(f"Memory after 'cat' (includes 'The'):\n{out[0, 1, :]}")
print(f"Memory after 'sat' (includes 'The' and 'cat'):\n{out[0, 2, :]}")

# Create a heatmap of the entire 'out' tensor (all 3 words)
entire_sentence_memory = out[0].detach().numpy() # Shape (3, 10)

plt.figure(figsize=(10, 4))
sns.heatmap(entire_sentence_memory, annot=True, cmap="coolwarm", 
            yticklabels=["After 'The'", "After 'cat'", "After 'sat'"])
plt.title("How the RNN Memory Builds Up Over Time")
plt.show()

# 2. Add the "Classifier" (The Decoder)
# This layer looks at the 10 columns and collapses them into 1 "Yes/No" score
classifier = nn.Linear(hidden_dim, 1)
prediction_score = classifier(hn.squeeze(0))
print("\nClassifier Result (The 'Interpretation'):")
print(f"Score: {prediction_score.item():.4f}")
print("Interpretation: Positive = Likely Animal, Negative = Likely Not Animal")




## TRAINING RNN
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# # 1. Prepare Synthetic Data (Sine Wave)
# steps = np.linspace(0, np.pi*2, 100)
# data = np.sin(steps)
# x = torch.Tensor(data[:-1]).view(-1, 1, 1) # Input: first 99 points
# y = torch.Tensor(data[1:]).view(-1, 1, 1)  # Target: shifted by 1 (next point)

# # 2. Define the RNN Model
# class SimpleRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleRNN, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x, h):
#         # out: output of the last layer for each time step
#         # h: the "memory" passed to the next step
#         out, h = self.rnn(x, h)
#         prediction = self.fc(out)
#         return prediction, h

# # 3. Initialize Model, Loss, and Optimizer
# model = SimpleRNN(1, 12, 1)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # 4. The Training Loop (Where learning happens)
# h = None  # Initial hidden state
# for epoch in range(200):
#     # Forward pass
#     prediction, h = model(x, h)
    
#     # We must detach the hidden state to prevent infinite backpropagation
#     h = h.detach() 
    
#     loss = criterion(prediction, y)
    
#     # Backward pass (Backpropagation Through Time)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if epoch % 50 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# print("Learning complete.")