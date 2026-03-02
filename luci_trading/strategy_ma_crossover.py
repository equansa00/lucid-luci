# strategy_ma_crossover.py
from luci_trading.data import get_stock_data


def ma_crossover_strategy(ticker, short_window=40, long_window=100):
    data = get_stock_data(ticker, period="1y")
    data['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    data['Signal'] = (data['short_mavg'] > data['long_mavg']).astype(int)
    return data
