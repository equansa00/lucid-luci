# scanner.py — SP500 top 50 by 1-month momentum
import sys
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from luci_trading.data import get_stock_data


# Hardcoded SP500 top 100 by market cap — avoids Wikipedia 403 bot blocking
_SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","AVGO","JPM",
    "LLY","UNH","V","XOM","MA","JNJ","PG","HD","COST","ABBV",
    "MRK","CVX","NFLX","CRM","BAC","ORCL","AMD","PEP","KO","WMT",
    "TMO","MCD","CSCO","ACN","ABT","LIN","DHR","TXN","ADBE","NEE",
    "PM","QCOM","GE","INTU","RTX","CAT","SPGI","HON","LOW","AMAT",
    "AMGN","T","ISRG","GS","IBM","ELV","BKNG","BLK","SYK","VRTX",
    "MDT","AXP","GILD","TJX","ADP","REGN","MMC","ETN","C","CB",
    "CI","MO","DUK","SO","BMY","NOW","PLD","SCHW","DE","ZTS",
    "PANW","BSX","SBUX","ADI","MDLZ","CME","ITW","EOG","SLB","AON",
    "WM","PH","FI","ICE","NOC","MCO","HUM","APH","CL","MSI",
]

def get_sp500_tickers() -> list[str]:
    return _SP500_TICKERS




def scan_sp500(top_n: int = 50) -> pd.DataFrame:
    """
    Fetch all SP500 tickers, rank by 1-month return, return top_n.
    """
    tickers = get_sp500_tickers()
    rows = []
    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period="1mo")
            if df.empty or len(df) < 2:
                continue
            ret = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
            rows.append({"ticker": ticker, "return_1mo": float(ret)})
        except Exception:
            continue

    result = (
        pd.DataFrame(rows)
        .sort_values("return_1mo", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index += 1  # 1-based rank
    return result


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"\nSP500 Top {n} by 1-month momentum\n{'─'*35}")
    df = scan_sp500(n)
    for _, row in df.iterrows():
        print(f"  {_:>3}. {row['ticker']:<7} {row['return_1mo']:+.2%}")
    print()
