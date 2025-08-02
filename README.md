# Stock Trading Strategy using Neural Network

This project uses a neural network (MLPClassifier) to predict stock trading signals based on technical indicators like RSI, MACD, and SuperTrend.


##  What the Project Does

- Downloads 15 years of stock data using yfinance.
- Calculates three trading indicators:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - SuperTrend
- Uses these to create Buy/Sell/Hold signals.
- Trains a Neural Network (MLPClassifier) on 10 years of data.
- Tests it on the next 5 years.
- Shows accuracy and classification report of the model.


