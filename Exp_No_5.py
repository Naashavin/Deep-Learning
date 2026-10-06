import numpy as np

def loss(w):
    return (w - 4) ** 2 + 3

def gradient(w):
    return 2 * (w - 4)

w = 10.0
lr = 0.1
epochs = 30

print(f"Initial weight: {w}, Initial Loss: {loss(w):.4f}\n")

for epoch in range(1, epochs + 1):
    grad = gradient(w)
    w = w - lr * grad
    current_loss = loss(w)
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:2d}: weight = {w:.4f}, loss = {current_loss:.4f}, grad = {grad:.4f}")

print(f"\nOptimal weight found: {w:.4f} (Target: 4.0)")
