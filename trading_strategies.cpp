#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

class TradingStrategies {
private:
    std::vector<double> prices;
    std::vector<double> highs;
    std::vector<double> lows;
    std::vector<double> closes;
    
public:
    TradingStrategies(const std::vector<double>& high_prices, 
                     const std::vector<double>& low_prices, 
                     const std::vector<double>& close_prices) 
        : highs(high_prices), lows(low_prices), closes(close_prices) {}
    
    // Calculate Simple Moving Average
    std::vector<double> sma(const std::vector<double>& data, int period) {
        std::vector<double> result(data.size(), 0.0);
        for (size_t i = period - 1; i < data.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += data[i - j];
            }
            result[i] = sum / period;
        }
        return result;
    }
    
    // Calculate Exponential Moving Average
    std::vector<double> ema(const std::vector<double>& data, int period) {
        std::vector<double> result(data.size(), 0.0);
        if (data.empty()) return result;
        
        double multiplier = 2.0 / (period + 1);
        result[0] = data[0];
        
        for (size_t i = 1; i < data.size(); ++i) {
            result[i] = (data[i] * multiplier) + (result[i-1] * (1 - multiplier));
        }
        return result;
    }
    
    // Calculate RSI
    std::vector<double> calculateRSI(int period = 14) {
        std::vector<double> rsi(closes.size(), 50.0);
        std::vector<double> gains, losses;
        
        for (size_t i = 1; i < closes.size(); ++i) {
            double change = closes[i] - closes[i-1];
            gains.push_back(change > 0 ? change : 0);
            losses.push_back(change < 0 ? -change : 0);
        }
        
        std::vector<double> avg_gains = ema(gains, period);
        std::vector<double> avg_losses = ema(losses, period);
        
        for (size_t i = period; i < closes.size(); ++i) {
            if (avg_losses[i-1] == 0) {
                rsi[i] = 100.0;
            } else {
                double rs = avg_gains[i-1] / avg_losses[i-1];
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        return rsi;
    }
    
    // Calculate MACD
    std::vector<std::vector<double>> calculateMACD(int fast_period = 12, int slow_period = 26, int signal_period = 9) {
        std::vector<double> ema_fast = ema(closes, fast_period);
        std::vector<double> ema_slow = ema(closes, slow_period);
        std::vector<double> macd_line(closes.size(), 0.0);
        
        for (size_t i = 0; i < closes.size(); ++i) {
            macd_line[i] = ema_fast[i] - ema_slow[i];
        }
        
        std::vector<double> signal_line = ema(macd_line, signal_period);
        std::vector<double> histogram(closes.size(), 0.0);
        
        for (size_t i = 0; i < closes.size(); ++i) {
            histogram[i] = macd_line[i] - signal_line[i];
        }
        
        return {macd_line, signal_line, histogram};
    }
    
    // Calculate SuperTrend
    std::vector<std::vector<double>> calculateSuperTrend(int period = 10, double multiplier = 3.0) {
        std::vector<double> hl2(closes.size());
        std::vector<double> atr(closes.size(), 0.0);
        std::vector<double> supertrend(closes.size(), 0.0);
        std::vector<double> trend(closes.size(), 1.0);
        
        // Calculate HL2 (typical price)
        for (size_t i = 0; i < closes.size(); ++i) {
            hl2[i] = (highs[i] + lows[i]) / 2.0;
        }
        
        // Calculate ATR
        std::vector<double> tr(closes.size(), 0.0);
        for (size_t i = 1; i < closes.size(); ++i) {
            double high_low = highs[i] - lows[i];
            double high_close = std::abs(highs[i] - closes[i-1]);
            double low_close = std::abs(lows[i] - closes[i-1]);
            tr[i] = std::max({high_low, high_close, low_close});
        }
        
        // Simple moving average of TR for ATR
        for (size_t i = period - 1; i < closes.size(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < period; ++j) {
                sum += tr[i - j];
            }
            atr[i] = sum / period;
        }
        
        // Calculate SuperTrend
        std::vector<double> upper_band(closes.size());
        std::vector<double> lower_band(closes.size());
        
        for (size_t i = 0; i < closes.size(); ++i) {
            upper_band[i] = hl2[i] + (multiplier * atr[i]);
            lower_band[i] = hl2[i] - (multiplier * atr[i]);
        }
        
        for (size_t i = 1; i < closes.size(); ++i) {
            // Determine trend
            if (closes[i] > upper_band[i-1]) {
                trend[i] = 1.0; // Uptrend
                supertrend[i] = lower_band[i];
            } else if (closes[i] < lower_band[i-1]) {
                trend[i] = -1.0; // Downtrend
                supertrend[i] = upper_band[i];
            } else {
                trend[i] = trend[i-1];
                if (trend[i] == 1.0) {
                    supertrend[i] = lower_band[i];
                } else {
                    supertrend[i] = upper_band[i];
                }
            }
        }
        
        return {supertrend, trend};
    }
    
    // RSI Strategy Signals
    std::vector<int> rsiStrategy(double oversold = 30.0, double overbought = 70.0) {
        std::vector<double> rsi = calculateRSI();
        std::vector<int> signals(closes.size(), 0);
        
        for (size_t i = 1; i < rsi.size(); ++i) {
            if (rsi[i-1] <= oversold && rsi[i] > oversold) {
                signals[i] = 1; // Buy signal
            } else if (rsi[i-1] >= overbought && rsi[i] < overbought) {
                signals[i] = -1; // Sell signal
            }
        }
        return signals;
    }
    
    // MACD Strategy Signals
    std::vector<int> macdStrategy() {
        auto macd_data = calculateMACD();
        std::vector<double> macd_line = macd_data[0];
        std::vector<double> signal_line = macd_data[1];
        std::vector<int> signals(closes.size(), 0);
        
        for (size_t i = 1; i < macd_line.size(); ++i) {
            if (macd_line[i-1] <= signal_line[i-1] && macd_line[i] > signal_line[i]) {
                signals[i] = 1; // Buy signal
            } else if (macd_line[i-1] >= signal_line[i-1] && macd_line[i] < signal_line[i]) {
                signals[i] = -1; // Sell signal
            }
        }
        return signals;
    }
    
    // SuperTrend Strategy Signals
    std::vector<int> supertrendStrategy() {
        auto st_data = calculateSuperTrend();
        std::vector<double> trend = st_data[1];
        std::vector<int> signals(closes.size(), 0);
        
        for (size_t i = 1; i < trend.size(); ++i) {
            if (trend[i-1] == -1.0 && trend[i] == 1.0) {
                signals[i] = 1; // Buy signal
            } else if (trend[i-1] == 1.0 && trend[i] == -1.0) {
                signals[i] = -1; // Sell signal
            }
        }
        return signals;
    }
    
    // Combined Strategy
    std::vector<int> combinedStrategy() {
        std::vector<int> rsi_sig = rsiStrategy();
        std::vector<int> macd_sig = macdStrategy();
        std::vector<int> st_sig = supertrendStrategy();
        std::vector<int> combined(closes.size(), 0);
        
        for (size_t i = 0; i < closes.size(); ++i) {
            int vote_sum = rsi_sig[i] + macd_sig[i] + st_sig[i];
            
            // Require at least 2 out of 3 indicators to agree
            if (vote_sum >= 2) {
                combined[i] = 1; // Buy
            } else if (vote_sum <= -2) {
                combined[i] = -1; // Sell
            }
        }
        return combined;
    }
    
    // Calculate strategy performance
    std::vector<double> calculatePerformance(const std::vector<int>& signals) {
        double total_return = 0.0;
        int num_trades = 0;
        int profitable_trades = 0;
        double position = 0.0;
        double entry_price = 0.0;
        
        for (size_t i = 1; i < signals.size(); ++i) {
            if (signals[i] == 1 && position == 0.0) { // Buy signal
                position = 1.0;
                entry_price = closes[i];
            } else if (signals[i] == -1 && position == 1.0) { // Sell signal
                double trade_return = (closes[i] - entry_price) / entry_price * 100.0;
                total_return += trade_return;
                num_trades++;
                if (trade_return > 0) profitable_trades++;
                position = 0.0;
            }
        }
        
        double success_rate = num_trades > 0 ? (double)profitable_trades / num_trades * 100.0 : 0.0;
        double avg_return = num_trades > 0 ? total_return / num_trades : 0.0;
        
        return {success_rate, avg_return, (double)num_trades};
    }
    
    // Get all strategy performances
    std::vector<std::vector<double>> getAllPerformances() {
        std::vector<int> rsi_signals = rsiStrategy();
        std::vector<int> macd_signals = macdStrategy();
        std::vector<int> st_signals = supertrendStrategy();
        std::vector<int> combined_signals = combinedStrategy();
        
        return {
            calculatePerformance(rsi_signals),
            calculatePerformance(macd_signals),
            calculatePerformance(st_signals),
            calculatePerformance(combined_signals)
        };
    }
};

namespace py = pybind11;

PYBIND11_MODULE(trading_strategies, m) {
    py::class_<TradingStrategies>(m, "TradingStrategies")
        .def(py::init<const std::vector<double>&, const std::vector<double>&, const std::vector<double>&>())
        .def("calculateRSI", &TradingStrategies::calculateRSI, py::arg("period") = 14)
        .def("calculateMACD", &TradingStrategies::calculateMACD, 
             py::arg("fast_period") = 12, py::arg("slow_period") = 26, py::arg("signal_period") = 9)
        .def("calculateSuperTrend", &TradingStrategies::calculateSuperTrend, 
             py::arg("period") = 10, py::arg("multiplier") = 3.0)
        .def("rsiStrategy", &TradingStrategies::rsiStrategy, 
             py::arg("oversold") = 30.0, py::arg("overbought") = 70.0)
        .def("macdStrategy", &TradingStrategies::macdStrategy)
        .def("supertrendStrategy", &TradingStrategies::supertrendStrategy)
        .def("combinedStrategy", &TradingStrategies::combinedStrategy)
        .def("calculatePerformance", &TradingStrategies::calculatePerformance)
        .def("getAllPerformances", &TradingStrategies::getAllPerformances);
}