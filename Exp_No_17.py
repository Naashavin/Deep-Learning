import torch
import torch.nn as nn
import torch.optim as optim

class SimpleBinaryCNN(nn.Module):
    def __init__(self):
        super(SimpleBinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.softmax(x)

X = torch.randn(100, 1, 28, 28)
y = torch.randint(0, 2, (100,))

model = SimpleBinaryCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(1, 11):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        print(f"Epoch [{epoch:2d}/10], Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    preds = torch.argmax(model(X), dim=1)
    acc = (preds == y).float().mean()
    print(f"\nFinal Binary CNN Softmax Accuracy: {acc.item() * 100:.2f}%")
