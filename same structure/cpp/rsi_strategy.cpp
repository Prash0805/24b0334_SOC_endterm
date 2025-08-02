#include <vector>

std::vector<int> compute_rsi_signals(const std::vector<double>& closes, int period = 14) {
    std::vector<int> signals(closes.size(), 0);  // 1 = buy, -1 = sell, 0 = hold
    for (size_t i = period; i < closes.size(); ++i) {
        double gain = 0, loss = 0;
        for (size_t j = i - period; j < i; ++j) {
            double change = closes[j + 1] - closes[j];
            if (change > 0) gain += change;
            else loss -= change;
        }
        double rs = (loss == 0) ? 100 : gain / loss;
        double rsi = 100 - (100 / (1 + rs));
        if (rsi < 30) signals[i] = 1;
        else if (rsi > 70) signals[i] = -1;
    }
    return signals;
}
