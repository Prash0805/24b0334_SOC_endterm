#ifndef SUPERTREND_STRATEGY_H
#define SUPERTREND_STRATEGY_H

#include "data_types.h"
#include <vector>

std::vector<TradeSignal> compute_supertrend_signals(const std::vector<OHLCV>& data, int period = 10, double multiplier = 3.0);

#endif
