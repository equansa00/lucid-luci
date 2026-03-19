"""
LUCI Polymarket Integration
Supports: browse markets, get prices, place orders, track positions
"""
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/beast/workspace/.env"))

HOST     = "https://clob.polymarket.com"
GAMMA    = "https://gamma-api.polymarket.com"
CHAIN_ID = 137

# ── Client factory ────────────────────────────────────────────────────────────

def get_client(authenticated=True):
    from py_clob_client.client import ClobClient
    if not authenticated:
        return ClobClient(HOST)
    pk      = os.getenv("POLYMARKET_PK")
    funder  = os.getenv("POLYMARKET_FUNDER")
    api_key = os.getenv("POLYMARKET_API_KEY")
    secret  = os.getenv("POLYMARKET_API_SECRET")
    phrase  = os.getenv("POLYMARKET_API_PASSPHRASE")
    if not pk:
        raise ValueError("POLYMARKET_PK not set in .env")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=1, funder=funder)
    if api_key:
        from py_clob_client.clob_types import ApiCreds
        client.set_api_creds(ApiCreds(api_key=api_key, api_secret=secret, api_passphrase=phrase))
    else:
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
    return client


# ── Market search ─────────────────────────────────────────────────────────────

def search_markets(query: str, limit: int = 10) -> list[dict]:
    """Search active markets by keyword via Gamma API."""
    url = f"{GAMMA}/markets"
    params = {"limit": limit, "active": "true", "closed": "false"}
    if query:
        params["_c"] = query  # Gamma uses _c for text search
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    markets = r.json()
    results = []
    for m in markets:
        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = []
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except Exception:
                prices = []
        results.append({
            "id":         m.get("id"),
            "question":   m.get("question", ""),
            "slug":       m.get("slug", ""),
            "volume":     float(m.get("volume", 0) or 0),
            "liquidity":  float(m.get("liquidity", 0) or 0),
            "outcomes":   outcomes,
            "prices":     prices,
            "end_date":   m.get("endDate", ""),
            "token_ids":  m.get("clobTokenIds", []),
        })
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results


def get_trending_markets(limit: int = 10) -> list[dict]:
    """Get top markets by volume."""
    url = f"{GAMMA}/markets"
    params = {"limit": limit, "active": "true", "closed": "false", "_sort": "volume"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Price data ────────────────────────────────────────────────────────────────

def get_market_price(token_id: str) -> dict:
    """Get bid/ask/mid for a token."""
    client = get_client(authenticated=False)
    try:
        bid = float(client.get_price(token_id, side="BUY")  or 0)
        ask = float(client.get_price(token_id, side="SELL") or 1)
        mid = round((bid + ask) / 2, 4)
        return {"bid": bid, "ask": ask, "mid": mid, "spread": round(ask - bid, 4)}
    except Exception as e:
        return {"error": str(e)}


# ── Positions ─────────────────────────────────────────────────────────────────

def get_positions() -> list[dict]:
    """Get open positions for the authenticated user."""
    funder = os.getenv("POLYMARKET_FUNDER", "")
    url = f"https://data-api.polymarket.com/positions?user={funder}&sizeThreshold=0.01"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        positions = []
        for p in data:
            positions.append({
                "market":    p.get("title", p.get("market", "")),
                "outcome":   p.get("outcome", ""),
                "size":      float(p.get("size", 0) or 0),
                "avg_price": float(p.get("avgPrice", 0) or 0),
                "cur_price": float(p.get("curPrice", 0) or 0),
                "value":     float(p.get("currentValue", 0) or 0),
                "pnl":       float(p.get("cashPnl", 0) or 0),
            })
        return positions
    except Exception as e:
        return [{"error": str(e)}]


def get_pnl() -> dict:
    """Get total P&L summary."""
    positions = get_positions()
    if positions and "error" in positions[0]:
        return positions[0]
    total_value = sum(p.get("value", 0) for p in positions)
    total_pnl   = sum(p.get("pnl", 0) for p in positions)
    return {
        "positions":    len(positions),
        "total_value":  round(total_value, 2),
        "total_pnl":    round(total_pnl, 2),
        "breakdown":    positions,
    }


# ── Order placement ───────────────────────────────────────────────────────────

def place_limit_order(token_id: str, price: float, size: float, side: str = "BUY") -> dict:
    """Place a limit order. side = 'BUY' or 'SELL'."""
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    client = get_client(authenticated=True)
    _side = BUY if side.upper() == "BUY" else SELL
    order = OrderArgs(token_id=token_id, price=price, size=size, side=_side)
    signed = client.create_order(order)
    resp = client.post_order(signed, OrderType.GTC)
    return resp


def place_market_order(token_id: str, amount: float, side: str = "BUY") -> dict:
    """Place a market order. amount = USDC amount to spend."""
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY, SELL
    client = get_client(authenticated=True)
    _side = BUY if side.upper() == "BUY" else SELL
    order = MarketOrderArgs(token_id=token_id, amount=amount, side=_side)
    signed = client.create_market_order(order)
    resp = client.post_order(signed, OrderType.FOK)
    return resp


# ── CLI / LUCI command handler ────────────────────────────────────────────────

def handle_command(cmd: str) -> str:
    """
    Parse and handle Polymarket commands from LUCI.
    Commands:
      /poly markets [query]
      /poly price <token_id>
      /poly positions
      /poly pnl
      /poly buy <token_id> <amount>
      /poly sell <token_id> <amount>
    """
    parts = cmd.strip().split()
    if len(parts) < 2:
        return _help()
    sub = parts[1].lower()

    if sub == "markets":
        query = " ".join(parts[2:]) if len(parts) > 2 else ""
        markets = search_markets(query, limit=8)
        if not markets:
            return "No markets found."
        lines = [f"{'MARKET':<55} {'YES':>6} {'NO':>6} {'VOL':>10}"]
        lines.append("─" * 85)
        for m in markets[:8]:
            q = m["question"][:53]
            prices = m.get("prices", [])
            yes_p = f"{float(prices[0])*100:.1f}¢" if prices else "  ?"
            no_p  = f"{float(prices[1])*100:.1f}¢" if len(prices) > 1 else "  ?"
            vol   = f"${m['volume']:,.0f}"
            lines.append(f"{q:<55} {yes_p:>6} {no_p:>6} {vol:>10}")
        return "\n".join(lines)

    elif sub == "price":
        if len(parts) < 3:
            return "Usage: /poly price <token_id>"
        data = get_market_price(parts[2])
        if "error" in data:
            return f"Error: {data['error']}"
        return (f"Bid:  {data['bid']*100:.2f}¢\n"
                f"Ask:  {data['ask']*100:.2f}¢\n"
                f"Mid:  {data['mid']*100:.2f}¢\n"
                f"Spread: {data['spread']*100:.2f}¢")

    elif sub == "positions":
        positions = get_positions()
        if not positions:
            return "No open positions."
        if "error" in positions[0]:
            return f"Error: {positions[0]['error']}"
        lines = []
        for p in positions:
            pnl_str = f"+${p['pnl']:.2f}" if p['pnl'] >= 0 else f"-${abs(p['pnl']):.2f}"
            lines.append(f"  {p['market'][:45]:<45} {p['outcome']:<4} "
                         f"size={p['size']:.2f} avg={p['avg_price']*100:.1f}¢ "
                         f"cur={p['cur_price']*100:.1f}¢ pnl={pnl_str}")
        return "\n".join(lines)

    elif sub == "pnl":
        data = get_pnl()
        if "error" in data:
            return f"Error: {data['error']}"
        pnl_str = f"+${data['total_pnl']:.2f}" if data['total_pnl'] >= 0 else f"-${abs(data['total_pnl']):.2f}"
        return (f"Open positions: {data['positions']}\n"
                f"Total value:    ${data['total_value']:.2f}\n"
                f"Total P&L:      {pnl_str}")

    elif sub in ("buy", "sell"):
        if len(parts) < 4:
            return f"Usage: /poly {sub} <token_id> <usdc_amount>"
        token_id = parts[2]
        try:
            amount = float(parts[3])
        except ValueError:
            return "Amount must be a number."
        confirm = input(f"  Place {sub.upper()} order: ${amount} USDC on token {token_id[:16]}...? [y/N] ")
        if confirm.lower() != "y":
            return "Order cancelled."
        try:
            resp = place_market_order(token_id, amount, side=sub.upper())
            return f"Order placed: {json.dumps(resp, indent=2)}"
        except Exception as e:
            return f"Order failed: {e}"

    else:
        return _help()


def _help() -> str:
    return """Polymarket commands:
  /poly markets [query]         — search markets
  /poly price <token_id>        — get bid/ask price
  /poly positions               — view open positions
  /poly pnl                     — P&L summary
  /poly buy <token_id> <$amt>   — place market buy
  /poly sell <token_id> <$amt>  — place market sell"""


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "/poly markets"
    print(handle_command("/poly " + " ".join(sys.argv[1:]) if not sys.argv[1:][0].startswith("/") else " ".join(sys.argv[1:])))
