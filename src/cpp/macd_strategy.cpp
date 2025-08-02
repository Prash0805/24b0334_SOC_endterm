#include "macd_strategy.h"
#include <vector>

double ema(const std::vector<double>& data, int start, int period) {
    double multiplier = 2.0 / (period + 1);
    double ema = data[start];
    for (int i = start + 1; i < data.size(); ++i)
        ema = (data[i] - ema) * multiplier + ema;
    return ema;
}

std::vector<TradeSignal> compute_macd_signals(const std::vector<OHLCV>& data) {
    std::vector<TradeSignal> signals;
    std::vector<double> close_prices;
    for (const auto& row : data) close_prices.push_back(row.close);

    for (size_t i = 0; i < data.size(); ++i) {
        if (i < 26) {
            signals.push_back({data[i].date, 0});
            continue;
        }

        double ema12 = ema(close_prices, i - 11, 12);
        double ema26 = ema(close_prices, i - 25, 26);
        double macd = ema12 - ema26;

        if (i < 35) {
            signals.push_back({data[i].date, 0});
            continue;
        }

        double signal_line = ema(close_prices, i - 8, 9);
        if (macd > signal_line)
            signals.push_back({data[i].date, 1});
        else if (macd < signal_line)
            signals.push_back({data[i].date, -1});
        else
            signals.push_back({data[i].date, 0});
    }

    return signals;
}
