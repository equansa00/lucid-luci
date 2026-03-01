import yfinance as yf
nvda = yf.Ticker('NVDA')
print(nvda.info['regularMarketPrice'])