"""
LUCI Trading — Alpaca paper execution layer.
All orders go to paper endpoint unless ALPACA_PAPER=false (never set this).
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_API_SECRET", "")
PAPER         = os.getenv("ALPACA_PAPER", "true").lower() != "false"


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=PAPER)


def alpaca_account() -> str:
    """Return a formatted account summary."""
    try:
        c = _client()
        a = c.get_account()
        buying_power = float(a.buying_power)
        cash         = float(a.cash)
        portfolio    = float(a.portfolio_value)
        pnl          = float(a.equity) - float(a.last_equity)
        pnl_sign     = "+" if pnl >= 0 else ""
        mode         = "📄 PAPER" if PAPER else "🔴 LIVE"
        return (
            f"{mode} | {a.status.value}\n"
            f"Portfolio:    ${portfolio:>12,.2f}\n"
            f"Cash:         ${cash:>12,.2f}\n"
            f"Buying Power: ${buying_power:>12,.2f}\n"
            f"Day P&L:      {pnl_sign}${pnl:,.2f}"
        )
    except Exception as e:
        return f"Account error: {e}"


def alpaca_positions() -> str:
    """Return all open positions."""
    try:
        c = _client()
        positions = c.get_all_positions()
        if not positions:
            return "No open positions."
        lines = ["Open positions:\n"]
        for p in positions:
            qty    = float(p.qty)
            price  = float(p.current_price)
            value  = float(p.market_value)
            pl     = float(p.unrealized_pl)
            pl_pct = float(p.unrealized_plpc) * 100
            sign   = "+" if pl >= 0 else ""
            lines.append(
                f"{p.symbol:6} {qty:>8.2f} shares @ ${price:.2f} "
                f"= ${value:,.2f}  ({sign}{pl_pct:.1f}%)"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Positions error: {e}"


def alpaca_orders(status: str = "open") -> str:
    """Return open or recent orders."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        c = _client()
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN if status == "open" else QueryOrderStatus.CLOSED, limit=10)
        orders = c.get_orders(filter=req)
        if not orders:
            return f"No {status} orders."
        lines = [f"{status.title()} orders:\n"]
        for o in orders:
            lines.append(
                f"{o.symbol:6} {o.side.value:4} {float(o.qty or 0):.2f} "
                f"@ {o.order_type.value}  [{o.status.value}]"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Orders error: {e}"


def alpaca_buy(symbol: str, qty: float, order_type: str = "market") -> dict:
    """
    Place a paper buy order. Returns dict with status and order info.
    order_type: 'market' or 'limit' (limit requires price kwarg — use market for now)
    """
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        c = _client()
        req = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = c.submit_order(req)
        return {
            "ok": True,
            "id": str(order.id),
            "symbol": order.symbol,
            "side": "BUY",
            "qty": float(order.qty),
            "type": order.order_type.value,
            "status": order.status.value,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def alpaca_sell(symbol: str, qty: float) -> dict:
    """Place a paper sell order."""
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        c = _client()
        req = MarketOrderRequest(
            symbol=symbol.upper(),
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = c.submit_order(req)
        return {
            "ok": True,
            "id": str(order.id),
            "symbol": order.symbol,
            "side": "SELL",
            "qty": float(order.qty),
            "type": order.order_type.value,
            "status": order.status.value,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def alpaca_cancel_all() -> str:
    """Cancel all open orders."""
    try:
        c = _client()
        cancelled = c.cancel_orders()
        return f"Cancelled {len(cancelled)} order(s)."
    except Exception as e:
        return f"Cancel error: {e}"


def alpaca_close_position(symbol: str) -> dict:
    """Close entire position for a symbol."""
    try:
        c = _client()
        order = c.close_position(symbol.upper())
        return {
            "ok": True,
            "symbol": symbol.upper(),
            "status": order.status.value,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
