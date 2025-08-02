import torch
import pandas as pd
import numpy as np
from model import TradeNet
from train import prepare_features

def evaluate_model(df, model_path):
    X, y_true = prepare_features(df)
    model = TradeNet(X.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        y_pred = model(X).argmax(dim=1).numpy()

    df['predicted'] = y_pred
    trades = df[df['predicted'] != 0].copy()
    trades['return'] = df['Close'].pct_change().shift(-1).fillna(0)

    successful_trades = trades[(trades['predicted'] == 1) & (trades['return'] > 0) |
                               (trades['predicted'] == 2) & (trades['return'] < 0)]

    success_rate = len(successful_trades) / len(trades) * 100 if len(trades) > 0 else 0
    avg_return = trades['return'].mean() * 100
    total_trades = len(trades)

    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Avg Return per Trade: {avg_return:.2f}%")
    print(f"Total Trades: {total_trades}")

    return {
        'Success Rate (%)': round(success_rate, 2),
        'Avg Return per Trade (%)': round(avg_return, 2),
        'Number of Trades': total_trades
    }
