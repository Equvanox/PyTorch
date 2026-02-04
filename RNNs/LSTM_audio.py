import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

# ---------------------------
# Load audio
# ---------------------------
# waveform, sr = torchaudio.load(
#     torchaudio.utils.download_asset(
#         "tutorial-assets/steam-train-whistle-daniel_simon.wav"
#     )
# )
# path = C:/Users/rajat.tawase/.cache/torch/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav

waveform, sr = torchaudio.load(r"D:\VisualStudioCode\Python\AI\DeepLearning\Pytorch\RNNs\file_example_WAV_1MG.wav")    # sr = sample rate

waveform = waveform.mean(dim=0)  # mono
frame_size = 400

frames = waveform.unfold(0, frame_size, frame_size)
frames = frames.unsqueeze(0)  # [1, seq_len, frame_size]

print("Frames shape:", frames.shape)

# ---------------------------
# Heatmap of input
# ---------------------------
plt.imshow(frames[0].T, aspect='auto', origin='lower')
plt.title("Audio Input Heatmap (Frames × Time)")
plt.xlabel("Time Steps")
plt.ylabel("Frame Samples")
plt.colorbar()
plt.show()

# ---------------------------
# LSTM Model
# ---------------------------
class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.lstm(x)

model = AudioLSTM(frame_size, 64)

output, (h_n, c_n) = model(frames)

print("Output shape:", output.shape)
print("Hidden shape:", h_n.shape)


# output from RNN: [batch, time, hidden]
hidden_activations = output[0].detach().numpy()

plt.imshow(hidden_activations.T, aspect='auto', cmap='viridis')
plt.xlabel("Time steps")
plt.ylabel("Hidden units")
plt.colorbar(label="Activation")
plt.title("LSTM Hidden State Heatmap")
plt.show()