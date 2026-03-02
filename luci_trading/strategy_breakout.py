# This file will contain the breakout trading strategy
import pandas as pd
from luci_trading.data import get_stock_data

def breakout_strategy(ticker, period='1mo', interval='1d'):
    data = get_stock_data(ticker, period=period, interval=interval)
    data['High_50'] = data['High'].rolling(window=50).max()
    data['Low_50'] = data['Low'].rolling(window=50).min()
    data['Signal'] = (data['Close'] > data['High_50']) | (data['Close'] < data['Low_50'])
    return data