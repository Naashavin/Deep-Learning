import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class FlexibleCNN(nn.Module):
    def __init__(self, activation_fn):
        super(FlexibleCNN, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act = activation_fn()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 2)

    def forward(self, x):
        x = self.pool(self.act(self.conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

X = torch.randn(128, 1, 28, 28)
y = torch.randint(0, 2, (128,))

configs = [
    {"batch_size": 16, "lr": 0.01, "opt": optim.SGD, "act": nn.ReLU},
    {"batch_size": 32, "lr": 0.001, "opt": optim.Adam, "act": nn.Tanh},
]

print(f"{'Batch Size':<10} | {'LR':<6} | {'Optimizer':<10} | {'Activation':<10} | {'Final Loss':<10}")
print("=" * 60)

for cfg in configs:
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    
    model = FlexibleCNN(cfg['act'])
    optimizer = cfg['opt'](model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    print(f"{cfg['batch_size']:<10} | {cfg['lr']:<6} | {cfg['opt'].__name__:<10} | {cfg['act'].__name__:<10} | {loss.item():.4f}")
