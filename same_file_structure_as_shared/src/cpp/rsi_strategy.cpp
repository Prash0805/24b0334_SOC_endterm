#include "rsi_strategy.h"
#include <vector>
#include <cmath>

std::vector<TradeSignal> compute_rsi_signals(const std::vector<OHLCV>& data, int period) {
    std::vector<TradeSignal> signals;
    std::vector<double> gains, losses;

    for (size_t i = 1; i < data.size(); ++i) {
        double change = data[i].close - data[i - 1].close;
        gains.push_back(change > 0 ? change : 0);
        losses.push_back(change < 0 ? -change : 0);
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (i < period) {
            signals.push_back({data[i].date, 0});
            continue;
        }

        double avg_gain = 0, avg_loss = 0;
        for (int j = i - period; j < i; ++j) {
            avg_gain += gains[j];
            avg_loss += losses[j];
        }
        avg_gain /= period;
        avg_loss /= period;

        double rs = avg_gain / (avg_loss + 1e-6);
        double rsi = 100 - (100 / (1 + rs));

        if (rsi > 70)
            signals.push_back({data[i].date, -1});
        else if (rsi < 30)
            signals.push_back({data[i].date, 1});
        else
            signals.push_back({data[i].date, 0});
    }

    return signals;
}
