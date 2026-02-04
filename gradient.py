import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# --- Dataset ---
X = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
Y = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

# --- Model Definition ---
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(1, 1, bias=False)  # 1 weight, no bias

    def forward(self, x):
        return self.fc(x)
    
# --- Complex Model: 2-layer MLP ---
class ComplexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([[2.0]]))   # 1x1
        self.w2 = nn.Parameter(torch.tensor([[1.0]]))   # 1x1
    def forward(self, x):
        # Nonlinear transformation to create complex terrain
        return torch.sin(self.w1 * x) + self.w2 * x**2

def compute_loss(w_val):
    with torch.no_grad():
        model.fc.weight.data.fill_(w_val)
        out = model(X)
        return criterion(out, Y).item()

model = SimpleNet()
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
# criterion = nn.HuberLoss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=0.1)
# optimizer = optim.Adam(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.2)
# optimizer = optim.Adagrad(model.parameters(), lr=0.1)

trajectory = []
for step in range(20):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, Y)
    loss.backward()
    optimizer.step()

    w = model.fc.weight.item()
    trajectory.append((w, 0, loss.item()))  # y=0 (dummy for 3D plot)
W1, W2 = np.meshgrid(np.linspace(-3, 5, 100), np.linspace(-1, 1, 2))  # W2 is dummy
Loss = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        Loss[i, j] = compute_loss(W1[i, j])
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8)
traj = np.array(trajectory)
ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='red', marker='o', label='Optimizer Path')
ax1.set_xlabel('Weight')
ax1.set_ylabel('Y (dummy)')
ax1.set_zlabel('Loss')
ax1.set_title('3D Loss Surface with Optimizer Path')
ax1.legend()
ax2 = fig.add_subplot(1, 2, 2)
w_values = np.linspace(-3, 5, 200)
loss_values = [compute_loss(w) for w in w_values]

ax2.plot(w_values, loss_values, label='Loss Curve')
ax2.plot(traj[:, 0], traj[:, 2], marker='o', color='red', label='Optimizer Path')
ax2.set_xlabel('Weight')
ax2.set_ylabel('Loss') 
ax2.set_title('2D Loss Curve with Optimizer Steps')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()