import csv
from typing import List, Tuple
from src.cpp.bindings import OHLCV

def load_ohlcv_from_csv(filepath: str) -> Tuple[List[OHLCV], List[str]]:
    ohlcv_data = []
    dates = []

    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ohlcv_data.append(OHLCV(
                date=row['Date'],
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume'])
            ))
            dates.append(row['Date'])

    return ohlcv_data, dates
