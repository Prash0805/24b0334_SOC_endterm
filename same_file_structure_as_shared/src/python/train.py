import torch
import torch.nn as nn
import torch.optim as optim
from model import TradeNet
import pandas as pd
import numpy as np

def prepare_features(df):
    # Use price deltas and indicators as inputs
    df['return'] = df['Close'].pct_change().fillna(0)
    X = df[['return', 'macd_signal', 'rsi_signal', 'supertrend_signal']].fillna(0).values
    y = df['label'].astype(int).values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_model(df, save_path):
    X, y = prepare_features(df)
    model = TradeNet(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(25):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/25, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
