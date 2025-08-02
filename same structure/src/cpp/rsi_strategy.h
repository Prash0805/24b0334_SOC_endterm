#ifndef RSI_STRATEGY_H
#define RSI_STRATEGY_H

#include "data_types.h"
#include <vector>

std::vector<TradeSignal> compute_rsi_signals(const std::vector<OHLCV>& data, int period = 14);

#endif
