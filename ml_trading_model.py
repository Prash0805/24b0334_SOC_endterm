import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Note: To use the C++ extension, compile it first with:
# pip install pybind11
# python setup.py build_ext --inplace
# Then uncomment the following line:
# import trading_strategies

class PythonTradingStrategies:
    """Python fallback implementation of trading strategies"""
    
    def __init__(self, high_prices, low_prices, close_prices):
        self.highs = np.array(high_prices)
        self.lows = np.array(low_prices)
        self.closes = np.array(close_prices)
    
    def sma(self, data, period):
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().fillna(0).values
    
    def ema(self, data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean().fillna(data[0]).values
    
    def calculateRSI(self, period=14):
        """Calculate RSI"""
        delta = np.diff(self.closes)
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        avg_gains = pd.Series(gains).ewm(span=period).mean()
        avg_losses = pd.Series(losses).ewm(span=period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi.fillna(50).values])
    
    def calculateMACD(self, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD"""
        ema_fast = self.ema(self.closes, fast_period)
        ema_slow = self.ema(self.closes, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return [macd_line, signal_line, histogram]
    
    def calculateSuperTrend(self, period=10, multiplier=3.0):
        """Calculate SuperTrend"""
        hl2 = (self.highs + self.lows) / 2
        
        # Calculate True Range
        high_low = self.highs - self.lows
        high_close = np.abs(self.highs - np.roll(self.closes, 1))
        low_close = np.abs(self.lows - np.roll(self.closes, 1))
        
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = high_low[0]  # First value
        
        # ATR
        atr = pd.Series(tr).rolling(window=period).mean().fillna(tr[0]).values
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = np.zeros_like(self.closes)
        trend = np.ones_like(self.closes)
        
        for i in range(1, len(self.closes)):
            if self.closes[i] > upper_band[i-1]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            elif self.closes[i] < lower_band[i-1]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = trend[i-1]
                supertrend[i] = lower_band[i] if trend[i] == 1 else upper_band[i]
        
        return [supertrend, trend]
    
    def rsiStrategy(self, oversold=30.0, overbought=70.0):
        """RSI Strategy Signals"""
        rsi = self.calculateRSI()
        signals = np.zeros_like(rsi, dtype=int)
        
        for i in range(1, len(rsi)):
            if rsi[i-1] <= oversold and rsi[i] > oversold:
                signals[i] = 1  # Buy
            elif rsi[i-1] >= overbought and rsi[i] < overbought:
                signals[i] = -1  # Sell
        
        return signals
    
    def macdStrategy(self):
        """MACD Strategy Signals"""
        macd_data = self.calculateMACD()
        macd_line, signal_line = macd_data[0], macd_data[1]
        signals = np.zeros_like(macd_line, dtype=int)
        
        for i in range(1, len(macd_line)):
            if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                signals[i] = 1  # Buy
            elif macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                signals[i] = -1  # Sell
        
        return signals
    
    def supertrendStrategy(self):
        """SuperTrend Strategy Signals"""
        st_data = self.calculateSuperTrend()
        trend = st_data[1]
        signals = np.zeros_like(trend, dtype=int)
        
        for i in range(1, len(trend)):
            if trend[i-1] == -1 and trend[i] == 1:
                signals[i] = 1  # Buy
            elif trend[i-1] == 1 and trend[i] == -1:
                signals[i] = -1  # Sell
        
        return signals
    
    def combinedStrategy(self):
        """Combined Strategy requiring 2/3 agreement"""
        rsi_sig = self.rsiStrategy()
        macd_sig = self.macdStrategy()
        st_sig = self.supertrendStrategy()
        combined = np.zeros_like(rsi_sig, dtype=int)
        
        for i in range(len(combined)):
            vote_sum = rsi_sig[i] + macd_sig[i] + st_sig[i]
            if vote_sum >= 2:
                combined[i] = 1  # Buy
            elif vote_sum <= -2:
                combined[i] = -1  # Sell
        
        return combined
    
    def calculatePerformance(self, signals):
        """Calculate strategy performance"""
        total_return = 0.0
        num_trades = 0
        profitable_trades = 0
        position = 0
        entry_price = 0.0
        
        for i in range(1, len(signals)):
            if signals[i] == 1 and position == 0:  # Buy signal
                position = 1
                entry_price = self.closes[i]
            elif signals[i] == -1 and position == 1:  # Sell signal
                trade_return = (self.closes[i] - entry_price) / entry_price * 100
                total_return += trade_return
                num_trades += 1
                if trade_return > 0:
                    profitable_trades += 1
                position = 0
        
        success_rate = (profitable_trades / num_trades * 100) if num_trades > 0 else 0
        avg_return = (total_return / num_trades) if num_trades > 0 else 0
        
        return [success_rate, avg_return, float(num_trades)]
    
    def getAllPerformances(self):
        """Get all strategy performances"""
        rsi_signals = self.rsiStrategy()
        macd_signals = self.macdStrategy()
        st_signals = self.supertrendStrategy()
        combined_signals = self.combinedStrategy()
        
        return [
            self.calculatePerformance(rsi_signals),
            self.calculatePerformance(macd_signals),
            self.calculatePerformance(st_signals),
            self.calculatePerformance(combined_signals)
        ]
    
    def nnStrategy(self, predictions):
        """Neural Network Strategy based on ML predictions"""
        signals = np.zeros_like(predictions, dtype=int)
        for i in range(len(predictions)):
            if predictions[i] == 1:
                signals[i] = 1  # Buy signal based on NN prediction
            else:
                signals[i] = -1 if i > 0 and signals[i-1] == 1 else 0  # Sell if we had a position
        return signals
    
class MLTradingModel:
    def __init__(self, symbol='AAPL', training_years=10, testing_years=5):
        self.symbol = symbol
        self.training_years = training_years
        self.testing_years = testing_years
        self.rf_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.data = None
        self.strategy_engine = None
        
    def fetch_data(self):
        """Fetch stock data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * (self.training_years + self.testing_years))
        
        print(f"Fetching {self.symbol} data from {start_date.date()} to {end_date.date()}")
        
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(start=start_date, end=end_date)
        
        if len(self.data) < 100:
            raise ValueError(f"Insufficient data for {self.symbol}")
        
        print(f"Data fetched: {len(self.data)} days")
        return self.data
    
    def prepare_features(self):
        """Prepare features for ML model"""
        # Initialize strategy engine
        self.strategy_engine = PythonTradingStrategies(
            self.data['High'].values,
            self.data['Low'].values,
            self.data['Close'].values
        )
        
        # Calculate indicators
        rsi = self.strategy_engine.calculateRSI()
        macd_data = self.strategy_engine.calculateMACD()
        st_data = self.strategy_engine.calculateSuperTrend()
        
        # Create feature dataframe
        features_df = pd.DataFrame({
            'RSI': rsi,
            'MACD': macd_data[0],
            'MACD_Signal': macd_data[1],
            'MACD_Histogram': macd_data[2],
            'SuperTrend': st_data[0],
            'SuperTrend_Direction': st_data[1],
            'Close': self.data['Close'].values,
            'Volume': self.data['Volume'].values,
            'High': self.data['High'].values,
            'Low': self.data['Low'].values
        })
        
        # Add price change features
        features_df['Price_Change'] = features_df['Close'].pct_change()
        features_df['Volume_MA'] = features_df['Volume'].rolling(20).mean()
        features_df['Volatility'] = features_df['Close'].rolling(20).std()
        
        # Create target variable (next day return > 0.5%)
        features_df['Target'] = (features_df['Close'].shift(-1) / features_df['Close'] - 1 > 0.005).astype(int)
        
        # Forward fill and backward fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        return features_df
    
    def split_data(self, features_df):
        """Split data into training and testing sets"""
        split_idx = len(features_df) - int(365 * self.testing_years)
        
        train_data = features_df.iloc[:split_idx].copy()
        test_data = features_df.iloc[split_idx:].copy()
        
        feature_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                       'SuperTrend_Direction', 'Price_Change', 'Volatility']
        
        X_train = train_data[feature_cols].values
        y_train = train_data['Target'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['Target'].values
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test, test_data
    
    def train_models(self, X_train, y_train):
        """Train both Random Forest and Neural Network models"""
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        
        print("Training Neural Network model...")
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        self.nn_model.fit(X_train, y_train)
        print("Models training completed!")
    
    def evaluate_strategies(self, test_data, nn_predictions=None):
        """Evaluate individual and combined strategies on test data"""
        # Create strategy engine for test data
        test_strategy_engine = PythonTradingStrategies(
            test_data['High'].values,
            test_data['Low'].values,
            test_data['Close'].values
        )
        
        # Get traditional strategy signals
        rsi_signals = test_strategy_engine.rsiStrategy()
        macd_signals = test_strategy_engine.macdStrategy()
        st_signals = test_strategy_engine.supertrendStrategy()
        combined_signals = test_strategy_engine.combinedStrategy()
        
        # Calculate performance for each strategy
        rsi_perf = test_strategy_engine.calculatePerformance(rsi_signals)
        macd_perf = test_strategy_engine.calculatePerformance(macd_signals)
        st_perf = test_strategy_engine.calculatePerformance(st_signals)
        combined_perf = test_strategy_engine.calculatePerformance(combined_signals)
        
        # Prepare results
        results = []
        
        # Individual strategies
        results.append({
            'Strategy': 'RSI Strategy',
            'Success Rate (%)': round(rsi_perf[0], 2),
            'Per-Trade Return (%)': round(rsi_perf[1], 2),
            'Total Trades': int(rsi_perf[2])
        })
        
        results.append({
            'Strategy': 'MACD Strategy', 
            'Success Rate (%)': round(macd_perf[0], 2),
            'Per-Trade Return (%)': round(macd_perf[1], 2),
            'Total Trades': int(macd_perf[2])
        })
        
        results.append({
            'Strategy': 'SuperTrend Strategy',
            'Success Rate (%)': round(st_perf[0], 2),
            'Per-Trade Return (%)': round(st_perf[1], 2),
            'Total Trades': int(st_perf[2])
        })
        
        results.append({
            'Strategy': 'Combined Strategy',
            'Success Rate (%)': round(combined_perf[0], 2),
            'Per-Trade Return (%)': round(combined_perf[1], 2),
            'Total Trades': int(combined_perf[2])
        })
        
        # Neural Network strategy if predictions provided
        if nn_predictions is not None:
            nn_signals = test_strategy_engine.nnStrategy(nn_predictions)
            nn_perf = test_strategy_engine.calculatePerformance(nn_signals)
            
            results.append({
                'Strategy': 'Neural Network Strategy',
                'Success Rate (%)': round(nn_perf[0], 2),
                'Per-Trade Return (%)': round(nn_perf[1], 2),
                'Total Trades': int(nn_perf[2])
            })
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, results_df, test_data, nn_predictions=None):
        """Create performance visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.symbol} Trading Strategies Performance Analysis', fontsize=16, fontweight='bold')
        
        # Success Rate comparison
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(results_df)]
        axes[0, 0].bar(range(len(results_df)), results_df['Success Rate (%)'], color=colors)
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_ylabel('Success Rate (%)')
        axes[0, 0].set_xticks(range(len(results_df)))
        axes[0, 0].set_xticklabels(results_df['Strategy'], rotation=45, ha='right')
        
        # Per-Trade Return comparison
        axes[0, 1].bar(range(len(results_df)), results_df['Per-Trade Return (%)'], color=colors)
        axes[0, 1].set_title('Average Per-Trade Return')
        axes[0, 1].set_ylabel('Per-Trade Return (%)')
        axes[0, 1].set_xticks(range(len(results_df)))
        axes[0, 1].set_xticklabels(results_df['Strategy'], rotation=45, ha='right')
        
        # Total Trades comparison
        axes[1, 0].bar(range(len(results_df)), results_df['Total Trades'], color=colors)
        axes[1, 0].set_title('Total Number of Trades')
        axes[1, 0].set_ylabel('Trade Count')
        axes[1, 0].set_xticks(range(len(results_df)))
        axes[1, 0].set_xticklabels(results_df['Strategy'], rotation=45, ha='right')
        
        # Price chart with signals
        test_strategy_engine = PythonTradingStrategies(
            test_data['High'].values,
            test_data['Low'].values,
            test_data['Close'].values
        )
        
        # Use Neural Network signals if available, otherwise combined strategy
        if nn_predictions is not None:
            signals = test_strategy_engine.nnStrategy(nn_predictions)
            strategy_name = "Neural Network"
        else:
            signals = test_strategy_engine.combinedStrategy()
            strategy_name = "Combined"
        
        # Plot price and signals
        axes[1, 1].plot(test_data.index, test_data['Close'], label='Close Price', linewidth=1)
        
        buy_signals = test_data.index[signals == 1]
        sell_signals = test_data.index[signals == -1]
        
        if len(buy_signals) > 0:
            axes[1, 1].scatter(buy_signals, test_data.loc[buy_signals, 'Close'], 
                              color='green', marker='^', s=60, label='Buy Signal', alpha=0.8)
        if len(sell_signals) > 0:
            axes[1, 1].scatter(sell_signals, test_data.loc[sell_signals, 'Close'], 
                              color='red', marker='v', s=60, label='Sell Signal', alpha=0.8)
        
        axes[1, 1].set_title(f'{strategy_name} Strategy Signals')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self):
        """Run complete analysis"""
        print("="*70)
        print(f"ML TRADING MODEL ANALYSIS FOR {self.symbol}")
        print("="*70)
        
        # Fetch and prepare data
        self.fetch_data()
        features_df = self.prepare_features()
        X_train, X_test, y_train, y_test, test_data = self.split_data(features_df)
        
        # Train models
        self.train_models(X_train, y_train)
        
        # Make predictions
        rf_pred = self.rf_model.predict(X_test)
        nn_pred = self.nn_model.predict(X_test)
        
        rf_accuracy = accuracy_score(y_test, rf_pred)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        
        print(f"\nRandom Forest Accuracy: {rf_accuracy:.3f}")
        print(f"Neural Network Accuracy: {nn_accuracy:.3f}")
        print(f"Training Period: {self.training_years} years")
        print(f"Testing Period: {self.testing_years} years")
        print(f"Test Data Points: {len(test_data)}")
        
        # Evaluate strategies (including NN)
        results_df = self.evaluate_strategies(test_data, nn_pred)
        
        print("\n" + "="*70)
        print("INDIVIDUAL STRATEGY PERFORMANCE (Testing Data Only)")
        print("="*70)
        
        # Print each strategy individually for clarity
        for _, row in results_df.iterrows():
            print(f"\n{row['Strategy'].upper()}:")
            print(f"  Success Rate: {row['Success Rate (%)']}%")
            print(f"  Per-Trade Return: {row['Per-Trade Return (%)']}%")
            print(f"  Total Trades: {row['Total Trades']}")
        
        print("\n" + "="*70)
        print("SUMMARY TABLE")
        print("="*70)
        print(results_df.to_string(index=False))
        
        # Feature importance
        feature_names = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 
                        'SuperTrend_Direction', 'Price_Change', 'Volatility']
        rf_feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'RF_Importance': self.rf_model.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        print("\n" + "="*50)
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print("="*50)
        print(rf_feature_importance.round(4).to_string(index=False))
        
        # Neural Network specific results
        nn_results = results_df[results_df['Strategy'] == 'Neural Network Strategy'].iloc[0]
        print("\n" + "="*50)
        print("NEURAL NETWORK STRATEGY RESULTS")
        print("="*50)
        print(f"Success Rate: {nn_results['Success Rate (%)']}%")
        print(f"Per-Trade Return: {nn_results['Per-Trade Return (%)']}%") 
        print(f"Total Trades Executed: {nn_results['Total Trades']}")
        
        # Create visualizations
        self.create_visualizations(results_df, test_data, nn_pred)
        
        # Additional analysis
        self.detailed_analysis(results_df, test_data)
        
        return results_df, rf_feature_importance
    
    def detailed_analysis(self, results_df, test_data):
        """Perform detailed analysis"""
        print("\n" + "="*40)
        print("DETAILED ANALYSIS")
        print("="*40)
        
        # Risk metrics
        test_returns = test_data['Close'].pct_change().dropna()
        volatility = test_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        max_drawdown = self.calculate_max_drawdown(test_data['Close'])
        
        print(f"Market Volatility (Annualized): {volatility:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        # Best performing strategy
        best_strategy = results_df.loc[results_df['Per-Trade Return (%)'].idxmax()]
        print(f"\nBest Strategy by Return: {best_strategy['Strategy']}")
        print(f"Return: {best_strategy['Per-Trade Return (%)']:.2f}%")
        print(f"Success Rate: {best_strategy['Success Rate (%)']:.2f}%")
        print(f"Total Trades: {best_strategy['Total Trades']}")
        
        # Most active strategy
        most_active = results_df.loc[results_df['Total Trades'].idxmax()]
        print(f"\nMost Active Strategy: {most_active['Strategy']}")
        print(f"Total Trades: {most_active['Total Trades']}")
        
        # Calculate Sharpe-like ratio for each strategy
        print(f"\nRisk-Adjusted Performance:")
        for _, row in results_df.iterrows():
            if row['Total Trades'] > 0:
                risk_adj_return = row['Per-Trade Return (%)'] / volatility * 100
                print(f"{row['Strategy']}: {risk_adj_return:.3f}")
    
    def calculate_max_drawdown(self, prices):
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak * 100
        return drawdown.min()

def main():
    """Main execution function"""
    # Initialize and run analysis
    model = MLTradingModel(symbol='AAPL', training_years=10, testing_years=5)
    
    try:
        results_df, feature_importance = model.run_analysis()
        
        print("\n" + "="*60)
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        
        # Find best overall strategy
        results_df['Score'] = (results_df['Success Rate (%)'] * 0.4 + 
                              results_df['Per-Trade Return (%)'] * 0.6)
        best_overall = results_df.loc[results_df['Score'].idxmax()]
        
        print(f"Recommended Strategy: {best_overall['Strategy']}")
        print(f"Success Rate: {best_overall['Success Rate (%)']:.2f}%")
        print(f"Avg Return: {best_overall['Per-Trade Return (%)']:.2f}%")
        print(f"Total Trades: {best_overall['Total Trades']}")
        
        print(f"\nKey Insights:")
        print(f"- Most important RF feature: {feature_importance.iloc[0]['Feature']}")
        print(f"- Neural Network achieved {nn_accuracy:.1%} prediction accuracy")
        print(f"- Analysis covers {model.testing_years} years of test data")
        
        # Final Neural Network summary
        nn_strategy_row = results_df[results_df['Strategy'] == 'Neural Network Strategy'].iloc[0]
        print(f"\n" + "="*50)
        print("FINAL NEURAL NETWORK STRATEGY SUMMARY")
        print("="*50)
        print(f"Success Rate: {nn_strategy_row['Success Rate (%)']}%")
        print(f"Per-Trade Return: {nn_strategy_row['Per-Trade Return (%)']}%")  
        print(f"Total Trades Executed: {nn_strategy_row['Total Trades']}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please ensure you have internet connection and required packages installed.")

if __name__ == "__main__":
    main()