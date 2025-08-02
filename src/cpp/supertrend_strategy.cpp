#include "supertrend_strategy.h"
#include <vector>
#include <cmath>

std::vector<TradeSignal> compute_supertrend_signals(const std::vector<OHLCV>& data, int period, double multiplier) {
    std::vector<TradeSignal> signals(data.size(), {"", 0});
    std::vector<double> atr(data.size(), 0);
    
    for (size_t i = 1; i < data.size(); ++i) {
        double tr = std::max({data[i].high - data[i].low,
                              std::abs(data[i].high - data[i-1].close),
                              std::abs(data[i].low - data[i-1].close)});
        atr[i] = (i < period) ? tr : (atr[i - 1] * (period - 1) + tr) / period;
    }

    bool in_uptrend = true;
    for (size_t i = period; i < data.size(); ++i) {
        double hl2 = (data[i].high + data[i].low) / 2;
        double upper_band = hl2 + multiplier * atr[i];
        double lower_band = hl2 - multiplier * atr[i];

        if (data[i].close > upper_band)
            in_uptrend = true;
        else if (data[i].close < lower_band)
            in_uptrend = false;

        int signal = in_uptrend ? 1 : -1;
        signals[i] = {data[i].date, signal};
    }

    return signals;
}
