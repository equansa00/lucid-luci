# This file will contain the momentum trading strategy
import pandas as pd
from luci_trading.data import get_stock_data

def momentum_strategy(ticker, period='1mo', interval='1d'):
    data = get_stock_data(ticker, period=period, interval=interval)
    data['Return'] = data['Close'].pct_change()
    data['Signal'] = data['Return'].rolling(window=20).mean() > 0
    return data