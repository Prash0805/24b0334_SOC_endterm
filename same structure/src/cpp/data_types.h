#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <vector>
#include <string>

struct OHLCV {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    double volume;
};

struct TradeSignal {
    std::string date;
    int signal; // 1 = buy, -1 = sell, 0 = hold
};

#endif // DATA_TYPES_H
