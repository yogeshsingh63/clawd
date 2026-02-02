#!/usr/bin/env python3
"""
BTC 15-Minute Scalp Bot (Supphieros-style)
==========================================
Strategy: Buy low-priced shares during high volatility when momentum aligns,
then flip quickly for small profits.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, List

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from btc_auto_trader import TAEngine, Candle, BTCAutoTrader, BTCMarket

load_dotenv()

# Configuration
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
SCAN_INTERVAL = int(os.getenv("SCALP_SCAN_INTERVAL", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Scalp settings
SCALP_ORDER_SIZE = float(os.getenv("SCALP_ORDER_SIZE", "50"))  # shares
SCALP_MAX_ENTRY_PRICE = float(os.getenv("SCALP_MAX_ENTRY_PRICE", "0.30"))  # dollars
SCALP_TAKE_PROFIT = float(os.getenv("SCALP_TAKE_PROFIT", "0.02"))  # dollars
SCALP_STOP_LOSS = float(os.getenv("SCALP_STOP_LOSS", "0.02"))  # dollars
SCALP_MIN_ALIGN = int(os.getenv("SCALP_MIN_ALIGN", "3"))  # signals aligned
SCALP_MIN_BTC_MOVE_PCT = float(os.getenv("SCALP_MIN_BTC_MOVE_PCT", "0.05"))  # %
SCALP_MAX_HOLD_SEC = int(os.getenv("SCALP_MAX_HOLD_SEC", "180"))
SCALP_FORCE_EXIT_MIN = float(os.getenv("SCALP_FORCE_EXIT_MIN", "1.5"))

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
BINANCE_API_URL = "https://api.binance.com/api/v3"
BTC_SERIES_ID = "10192"

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


class BTCScalpBot:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0)
        self.clob_client = None
        self.running = False
        self.current_market_id = None
        self.current_market = None
        self.ta_helper = BTCAutoTrader()
        self.position = None  # dict
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0

    async def initialize(self) -> bool:
        console.print(Panel.fit(
            "[bold blue]🎯 BTC Scalp Bot (High Volatility)[/bold blue]\n"
            f"Mode: {'[yellow]PAPER TRADING[/yellow]' if PAPER_TRADING else '[red]LIVE TRADING[/red]'}\n"
            f"Order Size: {SCALP_ORDER_SIZE} shares\n"
            f"Entry ≤ ${SCALP_MAX_ENTRY_PRICE:.2f} | TP +${SCALP_TAKE_PROFIT:.2f} | SL -${SCALP_STOP_LOSS:.2f}"
        ))

        if PAPER_TRADING:
            console.print("[yellow]📝 Paper trading mode[/yellow]")
        elif not PRIVATE_KEY:
            console.print("[red]❌ PRIVATE_KEY required for live trading[/red]")
            return False
        else:
            try:
                from py_clob_client.client import ClobClient
                self.clob_client = ClobClient(host=CLOB_API_URL, key=PRIVATE_KEY, chain_id=137)
                self.clob_client.set_api_creds(self.clob_client.derive_api_key())
                console.print("[green]✅ Connected to Polymarket[/green]")
            except Exception as e:
                console.print(f"[red]❌ Connection failed: {e}[/red]")
                return False

        # Reuse bot HTTP client for shared TA helper
        try:
            await self.ta_helper.http.aclose()
        except Exception:
            pass
        self.ta_helper.http = self.http

        return True

    async def fetch_binance_klines(self, interval: str = "1m", limit: int = 120) -> List[Candle]:
        try:
            resp = await self.http.get(
                f"{BINANCE_API_URL}/klines",
                params={"symbol": "BTCUSDT", "interval": interval, "limit": limit}
            )
            resp.raise_for_status()

            candles = []
            for k in resp.json():
                candles.append(Candle(
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5])
                ))
            return candles
        except Exception as e:
            logger.error(f"Binance error: {e}")
            return []

    async def fetch_btc_market(self):
        try:
            resp = await self.http.get(
                f"{GAMMA_API_URL}/events",
                params={"series_id": BTC_SERIES_ID, "active": "true", "closed": "false", "limit": "20"}
            )
            resp.raise_for_status()
            events = resp.json()

            if not events:
                return None

            now = datetime.now(timezone.utc)
            best_market = None
            best_end = None

            for event in events:
                for m in event.get("markets", []):
                    end_str = m.get("endDate")
                    if not end_str:
                        continue
                    try:
                        end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if end_time <= now:
                        continue
                    if best_end is None or end_time < best_end:
                        best_end = end_time
                        best_market = m

            if not best_market:
                return None

            outcomes = best_market.get("outcomes", [])
            if isinstance(outcomes, str):
                import json
                outcomes = json.loads(outcomes)
            clob_tokens = best_market.get("clobTokenIds", [])
            if isinstance(clob_tokens, str):
                import json
                clob_tokens = json.loads(clob_tokens)

            up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), None)
            down_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "down"), None)
            if up_idx is None or down_idx is None:
                return None

            up_token = clob_tokens[up_idx] if up_idx < len(clob_tokens) else None
            down_token = clob_tokens[down_idx] if down_idx < len(clob_tokens) else None
            if not up_token or not down_token:
                return None

            time_left = (best_end - now).total_seconds() / 60
            phase = "EARLY" if time_left > 10 else "MID" if time_left > 5 else "LATE"

            return {
                "market_id": str(best_market.get("id") or best_market.get("conditionId") or best_market.get("slug") or best_market.get("question", "")),
                "question": best_market.get("question", "")[:80],
                "end_time": best_end,
                "up_token_id": up_token,
                "down_token_id": down_token,
                "time_left_min": time_left,
                "phase": phase,
            }
        except Exception as e:
            logger.error(f"Market fetch error: {e}")
            return None

    async def get_book(self, token_id: str) -> Optional[dict]:
        try:
            resp = await self.http.get(f"{CLOB_API_URL}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def _fetch_prices(self, up_token: str, down_token: str):
        try:
            up_resp = await self.http.get(f"{CLOB_API_URL}/price", params={"token_id": up_token, "side": "buy"})
            down_resp = await self.http.get(f"{CLOB_API_URL}/price", params={"token_id": down_token, "side": "buy"})
            up_price = float(up_resp.json().get("price", 0)) * 100 if up_resp.status_code == 200 else None
            down_price = float(down_resp.json().get("price", 0)) * 100 if down_resp.status_code == 200 else None
            return up_price, down_price
        except Exception:
            return None, None

    def _avg_fill_from_asks(self, asks: List[dict], size: float) -> Optional[float]:
        if not asks:
            return None
        asks.sort(key=lambda x: float(x.get("price", 1.0)))
        remaining = size
        cost = 0.0
        for level in asks:
            price = float(level.get("price", 1.0))
            avail = float(level.get("size", 0.0))
            if avail <= 0:
                continue
            take = min(remaining, avail)
            cost += take * price
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            return None
        return cost / size if size > 0 else None

    def _avg_fill_from_bids(self, bids: List[dict], size: float) -> Optional[float]:
        if not bids:
            return None
        bids.sort(key=lambda x: float(x.get("price", 0.0)), reverse=True)
        remaining = size
        proceeds = 0.0
        for level in bids:
            price = float(level.get("price", 0.0))
            avail = float(level.get("size", 0.0))
            if avail <= 0:
                continue
            take = min(remaining, avail)
            proceeds += take * price
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            return None
        return proceeds / size if size > 0 else None

    async def get_avg_ask(self, token_id: str, size: float) -> Optional[float]:
        book = await self.get_book(token_id)
        if not book:
            return None
        return self._avg_fill_from_asks(book.get("asks", []), size)

    async def get_avg_bid(self, token_id: str, size: float) -> Optional[float]:
        book = await self.get_book(token_id)
        if not book:
            return None
        return self._avg_fill_from_bids(book.get("bids", []), size)

    def analyze_momentum(self, candles: List[Candle]) -> dict:
        closes = [c.close for c in candles]
        last_close = closes[-1] if closes else None
        prev_close = closes[-2] if len(closes) >= 2 else None

        rsi = TAEngine.compute_rsi(closes, 14)
        rsi_series = []
        for i in range(len(closes)):
            sub = closes[: i + 1]
            r = TAEngine.compute_rsi(sub, 14)
            if r is not None:
                rsi_series.append(r)
        rsi_slope = TAEngine.slope_last(rsi_series, 3)
        rsi_signal = "LONG" if rsi_slope and rsi_slope > 0 else "SHORT" if rsi_slope and rsi_slope < 0 else "NEUTRAL"

        macd = TAEngine.compute_macd(closes)
        if macd:
            macd_signal = "bullish" if macd["hist"] > 0 else "bearish" if macd["hist"] < 0 else "neutral"
        else:
            macd_signal = "neutral"

        ha = TAEngine.compute_heiken_ashi(candles)
        consec = TAEngine.count_consecutive(ha)
        heiken_color = consec["color"]
        heiken_count = consec["count"]

        delta_1m = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        delta_3m = closes[-1] - closes[-4] if len(closes) >= 4 else 0
        if delta_1m > 0 and delta_3m > 0:
            delta_signal = "LONG"
        elif delta_1m < 0 and delta_3m < 0:
            delta_signal = "SHORT"
        else:
            delta_signal = "NEUTRAL"

        vwap_series = TAEngine.compute_vwap_series(candles)
        vwap = vwap_series[-1] if vwap_series else None
        vwap_dist = ((last_close - vwap) / vwap) if (last_close is not None and vwap is not None and vwap != 0) else None

        momentum_long = 0
        momentum_short = 0

        if rsi_signal == "LONG":
            momentum_long += 1
        elif rsi_signal == "SHORT":
            momentum_short += 1

        if macd_signal == "bullish":
            momentum_long += 1
        elif macd_signal == "bearish":
            momentum_short += 1

        if heiken_color == "green" and heiken_count >= 2:
            momentum_long += 1
        elif heiken_color == "red" and heiken_count >= 2:
            momentum_short += 1

        if vwap_dist is not None:
            if vwap_dist > 0:
                momentum_long += 1
            elif vwap_dist < 0:
                momentum_short += 1

        if delta_signal == "LONG":
            momentum_long += 1
        elif delta_signal == "SHORT":
            momentum_short += 1

        if momentum_long > momentum_short:
            side = "UP"
            score = momentum_long
        elif momentum_short > momentum_long:
            side = "DOWN"
            score = momentum_short
        else:
            side = None
            score = 0

        vol_pct = 0.0
        if last_close and prev_close:
            vol_pct = abs(last_close - prev_close) / prev_close * 100

        return {
            "side": side,
            "score": score,
            "vol_pct": vol_pct,
            "delta_1m": delta_1m,
        }

    async def enter_position(self, market: dict, side: str, entry_price: float):
        token_id = market["up_token_id"] if side == "UP" else market["down_token_id"]
        target = min(0.99, entry_price + SCALP_TAKE_PROFIT)
        stop = max(0.01, entry_price - SCALP_STOP_LOSS)

        if PAPER_TRADING:
            self.position = {
                "side": side,
                "token_id": token_id,
                "entry_price": entry_price,
                "target_price": target,
                "stop_price": stop,
                "size": SCALP_ORDER_SIZE,
                "opened_at": datetime.now(timezone.utc),
            }
            self.trades += 1
            console.print(f"[yellow][PAPER] BUY {side} {SCALP_ORDER_SIZE} @ ${entry_price:.3f}[/yellow]")
            return

        try:
            from py_clob_client.clob_types import OrderType
            shares = SCALP_ORDER_SIZE
            order = self.clob_client.create_and_post_order(
                token_id=token_id, side="BUY", size=shares, price=entry_price, order_type=OrderType.FOK
            )
            if order and order.get("status") != "REJECTED":
                self.position = {
                    "side": side,
                    "token_id": token_id,
                    "entry_price": entry_price,
                    "target_price": target,
                    "stop_price": stop,
                    "size": SCALP_ORDER_SIZE,
                    "opened_at": datetime.now(timezone.utc),
                }
                self.trades += 1
                console.print(f"[green]✅ Bought {side} {SCALP_ORDER_SIZE} @ ${entry_price:.3f}[/green]")
        except Exception as e:
            logger.error(f"Trade error: {e}")

    async def exit_position(self, exit_price: float, reason: str):
        if not self.position:
            return
        side = self.position["side"]
        token_id = self.position["token_id"]
        size = self.position["size"]
        entry_price = self.position["entry_price"]

        if PAPER_TRADING:
            profit = (exit_price - entry_price) * size
            self.total_profit += profit
            if profit >= 0:
                self.wins += 1
            else:
                self.losses += 1
            console.print(f"[yellow][PAPER] SELL {side} {size} @ ${exit_price:.3f} ({reason})[/yellow]")
            console.print(f"[dim]P/L: {profit:+.2f} | Total: {self.total_profit:+.2f}[/dim]")
            self.position = None
            return

        try:
            from py_clob_client.clob_types import OrderType
            order = self.clob_client.create_and_post_order(
                token_id=token_id, side="SELL", size=size, price=exit_price, order_type=OrderType.FOK
            )
            if order and order.get("status") != "REJECTED":
                profit = (exit_price - entry_price) * size
                self.total_profit += profit
                if profit >= 0:
                    self.wins += 1
                else:
                    self.losses += 1
                console.print(f"[green]✅ Sold {side} {size} @ ${exit_price:.3f} ({reason})[/green]")
                self.position = None
        except Exception as e:
            logger.error(f"Sell error: {e}")

    async def run(self):
        if not await self.initialize():
            return

        self.running = True
        console.print("\n[bold green]🚀 Scalp bot running...[/bold green]\n")

        try:
            while self.running:
                console.print(f"\n{'='*50}")
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")

                market = await self.fetch_btc_market()
                candles = await self.fetch_binance_klines("1m", 120)

                if not market or not candles:
                    console.print("[yellow]Waiting for data...[/yellow]")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                if self.current_market_id and self.current_market_id != market["market_id"]:
                    if self.position:
                        bid = await self.get_avg_bid(self.position["token_id"], self.position["size"])
                        if bid is None:
                            bid = self.position["entry_price"]
                        await self.exit_position(bid, "new_round")
                    self.position = None

                self.current_market_id = market["market_id"]
                self.current_market = market["question"]

                console.print(f"Market: {market['question']}")
                console.print(f"Time Left: [bold]{market['time_left_min']:.1f} min[/bold] ({market['phase']})")

                up_price, down_price = await self._fetch_prices(market["up_token_id"], market["down_token_id"])
                if up_price is None:
                    up_price = 50.0
                if down_price is None:
                    down_price = 50.0
                console.print(f"Prices: [green]UP {up_price:.0f}¢[/green] | [red]DOWN {down_price:.0f}¢[/red]\n")

                market_snapshot = BTCMarket(
                    market_id=market["market_id"],
                    question=market["question"],
                    end_time=market["end_time"],
                    up_price=up_price,
                    down_price=down_price,
                    up_token_id=market["up_token_id"],
                    down_token_id=market["down_token_id"],
                    time_left_min=market["time_left_min"],
                    phase=market["phase"],
                )
                ta = self.ta_helper.analyze_ta(candles, market_snapshot)
                self.ta_helper.display_ta(ta, market_snapshot)

                if self.position:
                    bid = await self.get_avg_bid(self.position["token_id"], self.position["size"])
                    if bid is None:
                        await asyncio.sleep(SCAN_INTERVAL)
                        continue
                    hold_time = (datetime.now(timezone.utc) - self.position["opened_at"]).total_seconds()
                    if bid >= self.position["target_price"]:
                        await self.exit_position(bid, "take_profit")
                    elif bid <= self.position["stop_price"]:
                        await self.exit_position(bid, "stop_loss")
                    elif hold_time >= SCALP_MAX_HOLD_SEC or market["time_left_min"] <= SCALP_FORCE_EXIT_MIN:
                        await self.exit_position(bid, "time_exit")
                else:
                    m = self.analyze_momentum(candles)
                    if m["score"] >= SCALP_MIN_ALIGN and m["vol_pct"] >= SCALP_MIN_BTC_MOVE_PCT:
                        side = m["side"]
                        token_id = market["up_token_id"] if side == "UP" else market["down_token_id"]
                        ask = await self.get_avg_ask(token_id, SCALP_ORDER_SIZE)
                        if ask is not None and ask <= SCALP_MAX_ENTRY_PRICE:
                            await self.enter_position(market, side, ask)

                await asyncio.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped[/yellow]")
        finally:
            await self.http.aclose()
            try:
                await self.ta_helper.http.aclose()
            except Exception:
                pass


async def main():
    bot = BTCScalpBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
