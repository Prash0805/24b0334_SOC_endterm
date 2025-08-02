#include <vector>
#include <cmath>

std::vector<int> compute_supertrend_signals(const std::vector<double>& high,
                                            const std::vector<double>& low,
                                            const std::vector<double>& close,
                                            int period = 10, double multiplier = 3.0) {
    std::vector<int> signals(close.size(), 0);
    std::vector<double> atr(close.size(), 0);
    
    // ATR
    for (size_t i = 1; i < close.size(); ++i) {
        double tr = std::max({high[i] - low[i], fabs(high[i] - close[i-1]), fabs(low[i] - close[i-1])});
        atr[i] = (i < period) ? tr : ((atr[i-1] * (period - 1)) + tr) / period;
    }

    std::vector<double> upper(close.size()), lower(close.size());
    for (size_t i = period; i < close.size(); ++i) {
        double hl2 = (high[i] + low[i]) / 2.0;
        upper[i] = hl2 + multiplier * atr[i];
        lower[i] = hl2 - multiplier * atr[i];
        if (close[i] > upper[i - 1]) signals[i] = 1;
        else if (close[i] < lower[i - 1]) signals[i] = -1;
    }

    return signals;
}
                                                                                                                                                                        