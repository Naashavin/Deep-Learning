import torch
import torch.nn as nn
import numpy as np
from sklearn.neural_network import BernoulliRBM

X_rbm = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
rbm = BernoulliRBM(n_components=2, learning_rate=0.1, n_iter=50, random_state=42)
rbm.fit(X_rbm)
print("Bernoulli RBM Pseudo-Likelihood:", rbm.score_samples(X_rbm).mean())

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

rnn_model = SimpleRNN(input_size=5, hidden_size=10, output_size=2)
x_seq = torch.randn(8, 10, 5)
print("Simple RNN Output Shape:", rnn_model(x_seq).shape)

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm_model = SimpleLSTM(input_size=5, hidden_size=10, output_size=2)
print("Simple LSTM Output Shape:", lstm_model(x_seq).shape)
