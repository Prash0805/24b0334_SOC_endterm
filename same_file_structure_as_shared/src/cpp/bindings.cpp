#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "macd_strategy.h"
#include "rsi_strategy.h"
#include "supertrend_strategy.h"

namespace py = pybind11;

PYBIND11_MODULE(trading_strategies, m) {
    py::class_<OHLCV>(m, "OHLCV")
        .def(py::init<>())
        .def_readwrite("date", &OHLCV::date)
        .def_readwrite("open", &OHLCV::open)
        .def_readwrite("high", &OHLCV::high)
        .def_readwrite("low", &OHLCV::low)
        .def_readwrite("close", &OHLCV::close)
        .def_readwrite("volume", &OHLCV::volume);

    py::class_<TradeSignal>(m, "TradeSignal")
        .def_readwrite("date", &TradeSignal::date)
        .def_readwrite("signal", &TradeSignal::signal);

    m.def("compute_macd_signals", &compute_macd_signals);
    m.def("compute_rsi_signals", &compute_rsi_signals);
    m.def("compute_supertrend_signals", &compute_supertrend_signals);
}
