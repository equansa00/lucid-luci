# This file will contain data handling functions for the trading system
import yfinance as yf

def get_stock_data(ticker, period='1mo', interval='1d'):
    return yf.Ticker(ticker).history(period=period, interval=interval)

if __name__ == '__main__':
    print(get_stock_data('AAPL'))