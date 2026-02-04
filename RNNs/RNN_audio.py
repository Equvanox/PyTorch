import torch
import torch.nn as nn
import torchaudio


# ---------------------------
# 1. Load audio sample
# ---------------------------
# waveform, sample_rate = torchaudio.load(
#     torchaudio.utils.download_asset(
#         "tutorial-assets/steam-train-whistle-daniel_simon.wav"
#     )
# )
# path = C:/Users/rajat.tawase/.cache/torch/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav


waveform, sample_rate = torchaudio.load(r"D:\VisualStudioCode\Python\AI\DeepLearning\Pytorch\RNNs\file_example_WAV_1MG.wav")


# Waveform shape: [2, 109368] 
# [Channels (Stereo), Number of individual audio samples]
print("Waveform shape:", waveform.shape)

# Use mono (merging left and right channels)
waveform = waveform.mean(dim=0)  # [109368]

# Split into frames (chopping the long audio into readable segments)
frame_size = 400
frames = waveform.unfold(0, frame_size, frame_size) 
frames = frames.unsqueeze(0)  # Add batch dimension

# Frames shape: [1, 273, 400]
# [Batch size, Sequence length (number of frames), Features per frame]
print("Frames shape:", frames.shape)  

# ---------------------------
# 2. Audio RNN
# ---------------------------
class AudioRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # hidden_size=64 means the model has 64 "memory slots" or neurons
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.rnn(x)

model = AudioRNN(
    input_size=frame_size,
    hidden_size=64
)

# ---------------------------
# 3. Forward pass
# ---------------------------
output, hidden = model(frames)

# RNN output shape: [1, 273, 64]
# [Batch, Every time step, All 64 hidden states]
# This is a "diary" of what the RNN thought after hearing EACH of the 273 frames.
print("RNN output shape:", output.shape)

# Final hidden shape: [1, 1, 64]
# [Num_layers, Batch, Hidden_size]
# This is the "final summary" – the state of the memory after the VERY LAST frame.
print("Final hidden shape:", hidden.shape)