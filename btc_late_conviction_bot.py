#!/usr/bin/env python3
"""
BTC 15-Minute Late Conviction Bot (Bigsibas-style)
==================================================
Strategy: In the last 5-9 minutes, take larger size in the most
probable direction and hold to resolution.
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from btc_auto_trader import TAEngine, Candle, BTCAutoTrader

load_dotenv()

# Configuration
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
SCAN_INTERVAL = int(os.getenv("LATE_SCAN_INTERVAL", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
SETTLEMENT_GRACE_MIN = float(os.getenv("SETTLEMENT_GRACE_MIN", "3"))

# Late window settings
LATE_ENTRY_START = float(os.getenv("LATE_ENTRY_START", "9"))
LATE_ENTRY_END = float(os.getenv("LATE_ENTRY_END", "5"))
LATE_MIN_PROB = float(os.getenv("LATE_MIN_PROB", "0.58"))
LATE_ORDER_SIZE_USD = float(os.getenv("LATE_ORDER_SIZE_USD", "5.0"))
LATE_MAX_TRADES_PER_ROUND = int(os.getenv("LATE_MAX_TRADES_PER_ROUND", "2"))

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
BINANCE_API_URL = "https://api.binance.com/api/v3"
BTC_SERIES_ID = "10192"

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BTCMarket:
    market_id: str
    question: str
    end_time: datetime
    up_price: float
    down_price: float
    up_token_id: str
    down_token_id: str
    time_left_min: float
    phase: str


@dataclass
class RoundTrade:
    side: str
    bet: float
    price: float


@dataclass
class RoundRecord:
    market_id: str
    question: str
    end_time: datetime
    up_token_id: str
    down_token_id: str
    trades: List[RoundTrade]


class BTCLateConvictionBot:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0)
        self.clob_client = None
        self.running = False
        self.current_market_id = None
        self.current_market = None
        self.current_end_time = None
        self.current_up_token_id = None
        self.current_down_token_id = None
        self.ta_helper = BTCAutoTrader()
        self.round_trades: List[RoundTrade] = []
        self.pending_rounds: List[RoundRecord] = []
        self.trades_this_round = 0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0

    async def initialize(self) -> bool:
        console.print(Panel.fit(
            "[bold blue]🎯 BTC Late Conviction Bot[/bold blue]\n"
            f"Mode: {'[yellow]PAPER TRADING[/yellow]' if PAPER_TRADING else '[red]LIVE TRADING[/red]'}\n"
            f"Entry Window: {LATE_ENTRY_START:.0f}-{LATE_ENTRY_END:.0f} min left\n"
            f"Min Prob: {LATE_MIN_PROB:.2f} | Bet: ${LATE_ORDER_SIZE_USD:.2f}"
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

    async def fetch_binance_klines(self, interval: str = "1m", limit: int = 240) -> List[Candle]:
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

    async def fetch_btc_market(self) -> Optional[BTCMarket]:
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
                outcomes = json.loads(outcomes)
            clob_tokens = best_market.get("clobTokenIds", [])
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            outcome_prices = best_market.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                outcome_prices = json.loads(outcome_prices)

            up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), None)
            down_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "down"), None)

            if up_idx is None or down_idx is None:
                return None

            up_token = clob_tokens[up_idx] if up_idx < len(clob_tokens) else None
            down_token = clob_tokens[down_idx] if down_idx < len(clob_tokens) else None
            if not up_token or not down_token:
                return None

            up_price = float(outcome_prices[up_idx]) * 100 if up_idx < len(outcome_prices) else 50
            down_price = float(outcome_prices[down_idx]) * 100 if down_idx < len(outcome_prices) else 50

            time_left = (best_end - now).total_seconds() / 60
            phase = "EARLY" if time_left > 10 else "MID" if time_left > 5 else "LATE"
            market_id = best_market.get("id") or best_market.get("conditionId") or best_market.get("slug") or best_market.get("question", "")

            return BTCMarket(
                market_id=str(market_id),
                question=best_market.get("question", "")[:80],
                end_time=best_end,
                up_price=up_price,
                down_price=down_price,
                up_token_id=up_token,
                down_token_id=down_token,
                time_left_min=time_left,
                phase=phase,
            )
        except Exception as e:
            logger.error(f"Market fetch error: {e}")
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

    def _parse_json_list(self, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return []
        return value if isinstance(value, list) else []

    def _normalize_outcome(self, value, outcomes: List[str]):
        if value is None:
            return None
        if isinstance(value, dict):
            for k in ("outcome", "label", "name", "result", "answer"):
                if k in value:
                    value = value[k]
                    break
        if isinstance(value, (int, float)) and outcomes:
            idx = int(value)
            if 0 <= idx < len(outcomes):
                value = outcomes[idx]
        s = str(value).strip().lower()
        if s in ("up", "yes"):
            return "UP"
        if s in ("down", "no"):
            return "DOWN"
        return None

    async def _fetch_market_by_id(self, market_id: str) -> Optional[dict]:
        try:
            resp = await self.http.get(f"{GAMMA_API_URL}/markets/{market_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def _fetch_market_by_tokens(self, up_token: str, down_token: str) -> Optional[dict]:
        try:
            resp = await self.http.get(
                f"{GAMMA_API_URL}/events",
                params={"series_id": BTC_SERIES_ID, "limit": "40"}
            )
            resp.raise_for_status()
            events = resp.json()
            for event in events:
                for m in event.get("markets", []):
                    clob_tokens = self._parse_json_list(m.get("clobTokenIds", [])) or []
                    if up_token in clob_tokens and down_token in clob_tokens:
                        return m
            return None
        except Exception:
            return None

    def _extract_resolution(self, market: dict) -> Optional[dict]:
        outcomes = self._parse_json_list(market.get("outcomes", [])) or []
        outcome_prices = self._parse_json_list(market.get("outcomePrices", [])) or []

        up_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "up"), None)
        down_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == "down"), None)

        up_price = None
        down_price = None
        if up_idx is not None and up_idx < len(outcome_prices):
            try:
                up_price = float(outcome_prices[up_idx])
            except Exception:
                up_price = None
        if down_idx is not None and down_idx < len(outcome_prices):
            try:
                down_price = float(outcome_prices[down_idx])
            except Exception:
                down_price = None

        winner = None
        resolution_candidates = [
            market.get("resolvedOutcome"),
            market.get("resolutionOutcome"),
            market.get("finalOutcome"),
            market.get("winningOutcome"),
            market.get("outcome"),
            market.get("result"),
            market.get("answer"),
            market.get("resolution"),
            market.get("resolved"),
        ]
        if isinstance(market.get("resolution"), dict):
            resolution_candidates.append(market["resolution"].get("outcome"))
            resolution_candidates.append(market["resolution"].get("result"))

        for value in resolution_candidates:
            if value is not None:
                winner = self._normalize_outcome(value, outcomes)
                if winner:
                    break

        if winner is None and up_price is not None and down_price is not None:
            if up_price >= 0.99 and down_price <= 0.01:
                winner = "UP"
            elif down_price >= 0.99 and up_price <= 0.01:
                winner = "DOWN"

        return {
            "resolved": winner is not None,
            "winner": winner,
            "up_price": up_price,
            "down_price": down_price,
        }

    async def fetch_market_resolution(self, record: RoundRecord) -> Optional[dict]:
        market = None
        if record.market_id and str(record.market_id).isdigit():
            market = await self._fetch_market_by_id(record.market_id)
        if not market and record.up_token_id and record.down_token_id:
            market = await self._fetch_market_by_tokens(record.up_token_id, record.down_token_id)
        if not market:
            return None
        return self._extract_resolution(market)

    def _stash_current_round(self):
        if self.current_market_id and self.round_trades:
            self.pending_rounds.append(RoundRecord(
                market_id=self.current_market_id,
                question=self.current_market or "",
                end_time=self.current_end_time or datetime.now(timezone.utc),
                up_token_id=self.current_up_token_id or "",
                down_token_id=self.current_down_token_id or "",
                trades=list(self.round_trades),
            ))
        self.round_trades = []
        self.trades_this_round = 0
        self.current_market_id = None
        self.current_market = None
        self.current_end_time = None
        self.current_up_token_id = None
        self.current_down_token_id = None

    async def check_pending_settlements(self):
        if not self.pending_rounds:
            return
        still_pending: List[RoundRecord] = []
        for record in self.pending_rounds:
            now = datetime.now(timezone.utc)
            grace_ok = record.end_time and (now - record.end_time).total_seconds() / 60 >= SETTLEMENT_GRACE_MIN
            result = await self.fetch_market_resolution(record) if grace_ok else None
            if result and result.get("resolved") and result.get("winner"):
                self.settle_round(record, result)
            else:
                still_pending.append(record)
        self.pending_rounds = still_pending

    def settle_round(self, record: RoundRecord, result: dict):
        if not record.trades:
            return
        winner = result.get("winner")
        up_price = result.get("up_price")
        down_price = result.get("down_price")

        console.print(f"\n[bold cyan]{'═'*40}[/bold cyan]")
        console.print(f"[bold cyan]       ROUND SETTLED - {len(record.trades)} TRADES[/bold cyan]")
        console.print(f"[bold cyan]{'═'*40}[/bold cyan]")
        console.print(f"Market: {record.question}")

        if up_price is not None and down_price is not None:
            console.print(f"Final Prices: [green]UP {up_price*100:.1f}¢[/green] | [red]DOWN {down_price*100:.1f}¢[/red]")
        if winner:
            console.print(f"[bold]Winner: {winner}[/bold]")

        total_bet = 0.0
        total_payout = 0.0
        total_profit = 0.0

        for i, trade in enumerate(record.trades, 1):
            shares = trade.bet / trade.price
            payout = shares if trade.side == winner else 0.0
            profit = payout - trade.bet
            total_bet += trade.bet
            total_payout += payout
            total_profit += profit
            if profit > 0:
                self.wins += 1
            else:
                self.losses += 1
            console.print(f"[dim]#{i}[/dim] {trade.side} @ {trade.price*100:.1f}¢ | ${trade.bet:.2f} → {shares:.2f} shares | P/L {profit:+.2f}")

        self.total_profit += total_profit

        console.print(f"\n[bold]TOTAL BET: ${total_bet:.2f}[/bold]")
        console.print(f"[bold]TOTAL PAYOUT: ${total_payout:.2f}[/bold]")
        console.print(f"[bold]ROUND P/L: {total_profit:+.2f}[/bold]")
        console.print(f"[bold cyan]{'═'*40}[/bold cyan]\n")

    async def execute_trade(self, market: BTCMarket, side: str):
        if side == "UP":
            token_id = market.up_token_id
            price = market.up_price / 100
        else:
            token_id = market.down_token_id
            price = market.down_price / 100

        bet = LATE_ORDER_SIZE_USD
        if PAPER_TRADING:
            console.print(f"[yellow][PAPER] BUY {side} ${bet:.2f} @ {price*100:.1f}¢[/yellow]")
            self.round_trades.append(RoundTrade(side=side, bet=bet, price=price))
            self.trades += 1
            self.trades_this_round += 1
            return

        try:
            from py_clob_client.clob_types import OrderType
            shares = bet / price
            order = self.clob_client.create_and_post_order(
                token_id=token_id, side="BUY", size=shares, price=price, order_type=OrderType.FOK
            )
            if order and order.get("status") != "REJECTED":
                console.print(f"[green]✅ Bought {side} ${bet:.2f}[/green]")
                self.round_trades.append(RoundTrade(side=side, bet=bet, price=price))
                self.trades += 1
                self.trades_this_round += 1
        except Exception as e:
            logger.error(f"Trade error: {e}")

    def display_stats(self):
        profit_color = "[green]" if self.total_profit >= 0 else "[red]"
        profit_sign = "+" if self.total_profit >= 0 else ""
        pending = len(self.pending_rounds)
        pending_str = f" | Pending: {pending}" if pending else ""
        console.print(f"\n[dim]Trades: {self.trades} | W: {self.wins} L: {self.losses} | P/L: {profit_color}{profit_sign}${self.total_profit:.2f}[/]{pending_str}[/dim]")

    async def run(self):
        if not await self.initialize():
            return

        self.running = True
        console.print("\n[bold green]🚀 Late conviction bot running...[/bold green]\n")

        try:
            while self.running:
                console.print(f"\n{'='*50}")
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")

                market = await self.fetch_btc_market()
                candles = await self.fetch_binance_klines("1m", 240)

                if not market or not candles:
                    console.print("[yellow]Waiting for data...[/yellow]")
                    await self.check_pending_settlements()
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue

                if self.current_market_id and self.current_market_id != market.market_id:
                    self._stash_current_round()

                if not self.current_market_id:
                    self.current_market_id = market.market_id
                    self.current_market = market.question
                    self.current_end_time = market.end_time
                    self.current_up_token_id = market.up_token_id
                    self.current_down_token_id = market.down_token_id

                console.print(f"Market: {market.question}")
                console.print(f"Time Left: [bold]{market.time_left_min:.1f} min[/bold] ({market.phase})")

                up_price, down_price = await self._fetch_prices(market.up_token_id, market.down_token_id)
                if up_price is None:
                    up_price = market.up_price
                if down_price is None:
                    down_price = market.down_price
                console.print(f"Prices: [green]UP {up_price:.0f}¢[/green] | [red]DOWN {down_price:.0f}¢[/red]\n")

                market.up_price = up_price
                market.down_price = down_price

                ta = self.ta_helper.analyze_ta(candles, market)
                self.ta_helper.display_ta(ta, market)

                in_window = LATE_ENTRY_END <= market.time_left_min <= LATE_ENTRY_START
                if in_window and self.trades_this_round < LATE_MAX_TRADES_PER_ROUND:
                    closes = [c.close for c in candles]
                    vwap_series = TAEngine.compute_vwap_series(candles)
                    vwap = vwap_series[-1] if vwap_series else None
                    vwap_slope = None
                    if vwap_series and len(vwap_series) >= 5:
                        vwap_slope = (vwap_series[-1] - vwap_series[-5]) / 5 if vwap_series[-1] and vwap_series[-5] else None

                    rsi = TAEngine.compute_rsi(closes, 14)
                    rsi_series = []
                    for i in range(len(closes)):
                        sub = closes[: i + 1]
                        r = TAEngine.compute_rsi(sub, 14)
                        if r is not None:
                            rsi_series.append(r)
                    rsi_slope = TAEngine.slope_last(rsi_series, 3)
                    macd = TAEngine.compute_macd(closes)
                    ha = TAEngine.compute_heiken_ashi(candles)
                    consec = TAEngine.count_consecutive(ha)
                    last_price = closes[-1] if closes else None

                    scored = TAEngine.score_direction(
                        price=last_price,
                        vwap=vwap,
                        vwap_slope=vwap_slope,
                        rsi=rsi,
                        rsi_slope=rsi_slope,
                        macd=macd,
                        heiken_color=consec["color"],
                        heiken_count=consec["count"],
                        failed_vwap_reclaim=False,
                    )
                    time_aware = TAEngine.apply_time_awareness(scored["raw_up"], market.time_left_min, 15)
                    model_up = time_aware["adjusted_up"]
                    model_down = time_aware["adjusted_down"]

                    side = "UP" if model_up >= model_down else "DOWN"
                    best_prob = model_up if side == "UP" else model_down

                    console.print(f"[dim]Model probs: UP {model_up*100:.0f}% | DOWN {model_down*100:.0f}%[/dim]")

                    if best_prob >= LATE_MIN_PROB:
                        await self.execute_trade(market, side)
                    else:
                        console.print(f"[dim]No trade: prob {best_prob:.2f} < {LATE_MIN_PROB}[/dim]")
                elif not in_window:
                    console.print(f"[dim]Waiting for late window ({LATE_ENTRY_START:.0f}-{LATE_ENTRY_END:.0f} min left)[/dim]")
                else:
                    console.print("[dim]Trade limit reached this round[/dim]")

                self.display_stats()
                await self.check_pending_settlements()
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
    bot = BTCLateConvictionBot()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
