import pandas as pd
from python.data_processor import load_ohlcv_from_csv
from python.train import train_model
from python.test import evaluate_model

def generate_signals(ohlcv_data):
    macd = compute_macd_signals(ohlcv_data)
    rsi = compute_rsi_signals(ohlcv_data)
    supertrend = compute_supertrend_signals(ohlcv_data)
    return macd, rsi, supertrend

def build_dataframe(ohlcv_data, macd, rsi, supertrend):
    df = pd.DataFrame([{
        'Date': o.date,
        'Open': o.open,
        'High': o.high,
        'Low': o.low,
        'Close': o.close,
        'Volume': o.volume,
        'macd_signal': macd[i].signal,
        'rsi_signal': rsi[i].signal,
        'supertrend_signal': supertrend[i].signal
    } for i, o in enumerate(ohlcv_data)])

    # Label = 1 (buy), 2 (sell), 0 (hold)
    df['label'] = df[['macd_signal', 'rsi_signal', 'supertrend_signal']].mode(axis=1)[0].replace({-1:2}).fillna(0)
    return df

def main():
    train_ohlcv, _ = load_ohlcv_from_csv("data/AAPL_training.csv")
    test_ohlcv, _ = load_ohlcv_from_csv("data/AAPL_testing.csv")

    macd, rsi, supertrend = generate_signals(train_ohlcv)
    df_train = build_dataframe(train_ohlcv, macd, rsi, supertrend)
    train_model(df_train, "models/nn_model_weights.h5")

    macd_test, rsi_test, supertrend_test = generate_signals(test_ohlcv)
    df_test = build_dataframe(test_ohlcv, macd_test, rsi_test, supertrend_test)

    results = evaluate_model(df_test, "models/nn_model_weights.h5")
    pd.DataFrame([results]).to_csv("results/strategy_performance.csv", index=False)

if __name__ == "__main__":
    main()
