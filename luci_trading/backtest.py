# backtest.py — rank all 4 strategies by Sharpe ratio
import sys
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from luci_trading.strategy_momentum import momentum_strategy
from luci_trading.strategy_mean_reversion import mean_reversion_strategy
from luci_trading.strategy_breakout import breakout_strategy
from luci_trading.strategy_ma_crossover import ma_crossover_strategy


def _sharpe(signal_col, return_col, data, periods=252):
    """Annualised Sharpe ratio for a signal column."""
    pos = data[signal_col].shift(1).fillna(0)
    strat_ret = pos * data[return_col]
    mean = strat_ret.mean()
    std = strat_ret.std()
    if std == 0:
        return 0.0
    return float((mean / std) * np.sqrt(periods))


def run_backtest(ticker: str = "AAPL") -> list[dict]:
    """
    Run all 4 strategies on ticker, rank by Sharpe ratio.
    Returns list of dicts sorted best→worst.
    """
    results = []

    # Momentum
    df = momentum_strategy(ticker, period="2y")
    df['Return'] = df['Close'].pct_change().fillna(0)
    results.append({"strategy": "Momentum",       "sharpe": _sharpe("Signal", "Return", df)})

    # Mean reversion
    df = mean_reversion_strategy(ticker, period="2y")
    df['Return'] = df['Close'].pct_change().fillna(0)
    results.append({"strategy": "Mean Reversion", "sharpe": _sharpe("Signal", "Return", df)})

    # Breakout
    df = breakout_strategy(ticker, period="2y")
    df['Return'] = df['Close'].pct_change().fillna(0)
    results.append({"strategy": "Breakout",       "sharpe": _sharpe("Signal", "Return", df)})

    # MA Crossover
    df = ma_crossover_strategy(ticker)
    df['Return'] = df['Close'].pct_change().fillna(0)
    results.append({"strategy": "MA Crossover",   "sharpe": _sharpe("Signal", "Return", df)})

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"\nBacktest: {ticker}\n{'─'*35}")
    for rank, r in enumerate(run_backtest(ticker), 1):
        print(f"  {rank}. {r['strategy']:<18} Sharpe: {r['sharpe']:+.3f}")
    print()
