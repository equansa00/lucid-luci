"""
luci_trading_agent.py — LUCI Trading Integration Layer

Wraps luci_trading (scanner, backtest, strategies) and exposes
clean functions for Telegram commands and the morning briefing.

Paper trading only until explicitly switched to live.
All decisions logged to luci_trade_log.db (SQLite).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── path setup ──────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

# ── config ──────────────────────────────────────────────────────
TRADE_LOG_DB  = WORKSPACE / "luci_trade_log.db"
MAX_SCAN_SHOW = int(os.getenv("LUCI_TRADE_SCAN_TOP", "10"))
PAPER_MODE    = os.getenv("LUCI_TRADE_LIVE", "0") != "1"   # default: paper

# ── DB setup ────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(TRADE_LOG_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            ticker      TEXT,
            strategy    TEXT,
            details     TEXT,
            mode        TEXT    DEFAULT 'paper'
        )
    """)
    con.commit()
    return con


def _log(action: str, ticker: str = "", strategy: str = "", details: Any = None) -> None:
    """Log every decision to SQLite."""
    try:
        con = _db()
        con.execute(
            "INSERT INTO trade_log (ts, action, ticker, strategy, details, mode) VALUES (?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                action,
                ticker,
                strategy,
                json.dumps(details) if details is not None else "",
                "paper" if PAPER_MODE else "live",
            ),
        )
        con.commit()
        con.close()
    except Exception as e:
        print(f"[trade_log] write error: {e}", file=sys.stderr)


# ── Public API ───────────────────────────────────────────────────

def trade_scan(top_n: int = MAX_SCAN_SHOW) -> str:
    """
    Run the SP500 momentum scanner and return a formatted string.
    Safe to call from Telegram handler — catches all errors.
    """
    try:
        from luci_trading.scanner import scan_sp500
        df = scan_sp500(top_n)
        if df.empty:
            return "⚠️ Scanner returned no results. Market may be closed or data unavailable."

        lines = [f"📈 *Top {top_n} SP500 Momentum*\n"]
        for rank, row in df.iterrows():
            arrow = "▲" if row["return_1mo"] > 0 else "▼"
            lines.append(f"  {rank:>2}. {row['ticker']:<7} {arrow} {row['return_1mo']:+.2%}")

        lines.append(f"\n_{'Paper mode' if PAPER_MODE else '🔴 LIVE'} · {datetime.now().strftime('%b %d %H:%M')}_")
        result = "\n".join(lines)
        _log("scan", details={"top_n": top_n, "results": len(df)})
        return result
    except Exception as e:
        _log("scan_error", details={"error": str(e)})
        return f"❌ Scan error: {e}"


def trade_backtest(ticker: str) -> str:
    """
    Run all 4 strategies on ticker, return ranked results as string.
    """
    ticker = ticker.upper().strip()
    try:
        from luci_trading.backtest import run_backtest
        results = run_backtest(ticker)
        if not results:
            return f"⚠️ No backtest results for {ticker}."

        lines = [f"📊 *Backtest: {ticker}*\n"]
        medals = ["🥇", "🥈", "🥉", "  4."]
        for i, r in enumerate(results):
            sharpe = r["sharpe"]
            bar    = "█" * max(0, min(10, int(abs(sharpe) * 4)))
            color  = "+" if sharpe > 0 else ""
            lines.append(f"  {medals[i]} {r['strategy']:<18} Sharpe {color}{sharpe:.3f}  {bar}")

        best = results[0]
        lines.append(f"\n_Best strategy: {best['strategy']} (Sharpe {best['sharpe']:+.3f})_")
        result = "\n".join(lines)
        _log("backtest", ticker=ticker, strategy=best["strategy"], details=results)
        return result
    except Exception as e:
        _log("backtest_error", ticker=ticker, details={"error": str(e)})
        return f"❌ Backtest error for {ticker}: {e}"


def trade_status() -> str:
    """
    Show paper trading status — recent log entries and mode.
    """
    try:
        con  = _db()
        rows = con.execute(
            "SELECT ts, action, ticker, strategy, mode FROM trade_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        con.close()

        mode_line = f"{'📄 Paper mode' if PAPER_MODE else '🔴 LIVE MODE'}"
        if not rows:
            return f"📋 *Trading Status*\n\n{mode_line}\n\nNo activity logged yet.\nRun `/trade scan` to start."

        lines = [f"📋 *Trading Status* — {mode_line}\n"]
        for ts, action, ticker, strategy, mode in rows:
            t = ts[:16].replace("T", " ")
            parts = [f"  `{t}`", action]
            if ticker:
                parts.append(ticker)
            if strategy:
                parts.append(f"→ {strategy}")
            lines.append(" ".join(parts))

        return "\n".join(lines)
    except Exception as e:
        return f"❌ Status error: {e}"


def trade_pulse() -> str:
    """
    Quick market pulse — top 3 movers + best strategy on SPY.
    Designed for the morning briefing.
    """
    lines = []
    try:
        from luci_trading.scanner import scan_sp500
        df = scan_sp500(5)
        if not df.empty:
            lines.append("📈 *Top Momentum*")
            for rank, row in df.head(3).iterrows():
                arrow = "▲" if row["return_1mo"] > 0 else "▼"
                lines.append(f"  {row['ticker']:<7} {arrow} {row['return_1mo']:+.2%}")
    except Exception as e:
        lines.append(f"  Scanner unavailable: {e}")

    try:
        from luci_trading.backtest import run_backtest
        results = run_backtest("SPY")
        if results:
            best = results[0]
            lines.append(f"\n🎯 Best strategy on SPY: *{best['strategy']}* (Sharpe {best['sharpe']:+.3f})")
    except Exception as e:
        lines.append(f"\n  Backtest unavailable: {e}")

    lines.append(f"\n_{'Paper' if PAPER_MODE else '🔴 LIVE'} · {datetime.now().strftime('%b %d %H:%M')}_")
    _log("pulse")
    return "\n".join(lines) if lines else "Trading pulse unavailable."


def trade_briefing_block() -> str:
    """
    Compact block for the morning briefing.
    Returns empty string on any error so briefing always sends.
    """
    try:
        return "\n\n📊 *Trading Pulse*\n" + trade_pulse()
    except Exception:
        return ""
