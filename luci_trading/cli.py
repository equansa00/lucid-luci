#!/usr/bin/env python3
"""
luci-trader — LUCI stock trading CLI
Commands:
  scan              SP500 top 50 by 1-month momentum
  backtest TICKER   Rank all 4 strategies on TICKER by Sharpe ratio
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_scan(args):
    from luci_trading.scanner import scan_sp500
    n = args.top
    print(f"\nSP500 Top {n} by 1-month momentum\n{'─'*40}")
    df = scan_sp500(n)
    for rank, row in df.iterrows():
        print(f"  {rank:>3}. {row['ticker']:<7}  {row['return_1mo']:+.2%}")
    print()


def cmd_backtest(args):
    from luci_trading.backtest import run_backtest
    ticker = args.ticker.upper()
    print(f"\nBacktest: {ticker}\n{'─'*40}")
    for rank, r in enumerate(run_backtest(ticker), 1):
        bar = "█" * max(0, int(r["sharpe"] * 5))
        print(f"  {rank}. {r['strategy']:<18}  Sharpe {r['sharpe']:+.3f}  {bar}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="luci-trader",
        description="LUCI stock trading system"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="SP500 top 50 momentum scan")
    p_scan.add_argument("--top", type=int, default=50, help="How many to show (default 50)")
    p_scan.set_defaults(func=cmd_scan)

    p_bt = sub.add_parser("backtest", help="Backtest all strategies on a ticker")
    p_bt.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    p_bt.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
