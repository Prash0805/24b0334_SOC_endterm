import torch
import torch.nn as nn

class TradeNet(nn.Module):
    def __init__(self, input_dim):
        super(TradeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 0 = hold, 1 = buy, 2 = sell
        )

    def forward(self, x):
        return self.model(x)
