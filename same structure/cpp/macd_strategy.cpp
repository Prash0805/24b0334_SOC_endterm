#include <vector>

std::vector<int> compute_macd_signals(const std::vector<double>& closes) {
    int short_period = 12, long_period = 26, signal_period = 9;
    std::vector<double> macd, signal_line;
    std::vector<int> signals(closes.size(), 0);

    // Compute EMA helper
    auto ema = [&](int period, size_t i) {
        double k = 2.0 / (period + 1);
        double ema = closes[i - period];
        for (size_t j = i - period + 1; j <= i; ++j)
            ema = closes[j] * k + ema * (1 - k);
        return ema;
    };

    for (size_t i = long_period; i < closes.size(); ++i) {
        double macd_val = ema(short_period, i) - ema(long_period, i);
        macd.push_back(macd_val);
        if (i >= long_period + signal_period) {
            double signal = 0;
            for (int j = 0; j < signal_period; ++j)
                signal += macd[macd.size() - j - 1];
            signal /= signal_period;
            signal_line.push_back(signal);
            if (macd_val > signal) signals[i] = 1;
            else if (macd_val < signal) signals[i] = -1;
        }
    }

    return signals;
}
