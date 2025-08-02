cmake_minimum_required(VERSION 3.14)
project(TradingStrategies)
set(CMAKE_CXX_STANDARD 17)
find_package(pybind11 REQUIRED)
include_directories(include)
add_library(strategies SHARED
    cpp/macd_strategy.cpp
    cpp/rsi_strategy.cpp
    cpp/supertrend_strategy.cpp
    cpp/trading_bindings.cpp
    cpp/utils.cpp
)
target_link_libraries(strategies PRIVATE pybind11::module)