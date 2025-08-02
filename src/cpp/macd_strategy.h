#ifndef MACD_STRATEGY_H
#define MACD_STRATEGY_H

#include "data_types.h"
#include <vector>

std::vector<TradeSignal> compute_macd_signals(const std::vector<OHLCV>& data);

#endif
