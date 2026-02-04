import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt

# ---------------------------
# Load audio
# ---------------------------
waveform, sr = torchaudio.load(
    torchaudio.utils.download_asset(
        "tutorial-assets/steam-train-whistle-daniel_simon.wav"
    )
)

waveform = waveform.mean(dim=0)
frame_size = 400

frames = waveform.unfold(0, frame_size, frame_size)
frames = frames.unsqueeze(0)

# ---------------------------
# Input heatmap
# ---------------------------
plt.imshow(frames[0].T, aspect='auto', origin='lower')
plt.title("Audio Input Heatmap")
plt.xlabel("Time Steps")
plt.ylabel("Frame Samples")
plt.colorbar()
plt.show()

# ---------------------------
# GRU Model
# ---------------------------
class AudioGRU(nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)

    def forward(self, x):
        return self.gru(x)

model = AudioGRU(frame_size, 64)

output, h_n = model(frames)

print("GRU Output shape:", output.shape)

# ---------------------------
# Output heatmap
# ---------------------------
plt.imshow(output[0].detach().T, aspect='auto', origin='lower')
plt.title("GRU Output Heatmap (Hidden × Time)")
plt.xlabel("Time Steps")
plt.ylabel("Hidden Units")
plt.colorbar()
plt.show()
