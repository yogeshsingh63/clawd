#!/usr/bin/env python3
"""
BTC 15-Minute Auto-Trader with TA Indicators
=============================================
Uses proper Technical Analysis indicators like the original btc_assistant.

Trading Rules:
1. At least 3 indicators must align (TA, RSI, MACD, Heiken, Delta)
2. Extreme RSI (< 25 or > 75) = likely reversal, don't chase
3. Delta 1m/3m confirms signals
4. Entry timing: 5-10 min = best entry zone
5. Skip if confidence < 60%
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

load_dotenv()

# Configuration
PRIVATE_KEY = os.getenv("PRIVATE_KEY", "")
MAX_BET_SIZE = float(os.getenv("MAX_BET_SIZE", "1.00"))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
MAX_TOTAL_EXPOSURE = float(os.getenv("MAX_TOTAL_EXPOSURE", "3"))
MIN_CONFIDENCE = float(os.getenv("BTC_MIN_CONFIDENCE", "0.60"))  # 60% per rules
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "10"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Trading window within the 15-min market (in minutes left)
# e.g., TRADE_WINDOW_START=10, TRADE_WINDOW_END=2 means trade when 10-2 min left
TRADE_WINDOW_START = float(os.getenv("TRADE_WINDOW_START", "15"))  # Start trading when X min left
TRADE_WINDOW_END = float(os.getenv("TRADE_WINDOW_END", "0"))  # Stop trading when X min left

# API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"
BINANCE_API_URL = "https://api.binance.com/api/v3"
BTC_SERIES_ID = "10192"

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TASignals:
    """Technical Analysis signals"""
    # Heiken Ashi
    heiken_color: str  # "green" or "red"
    heiken_count: int  # consecutive candles
    
    # RSI
    rsi: float
    rsi_signal: str  # "LONG", "SHORT", "NEUTRAL"
    
    # MACD
    macd_signal: str  # "bullish", "bearish", "neutral"
    macd_expanding: bool
    
    # Delta
    delta_1m: float
    delta_3m: float
    delta_signal: str  # "LONG", "SHORT", "NEUTRAL"
    
    # Overall
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
    indicators_aligned: int  # count of aligned indicators
    trade_signal: bool


@dataclass
class BTCMarket:
    question: str
    end_time: datetime
    up_price: float
    down_price: float
    up_token_id: str
    down_token_id: str
    time_left_min: float
    phase: str


class TAEngine:
    """Technical Analysis Engine - same logic as btc_assistant"""
    
    @staticmethod
    def compute_rsi(closes: List[float], period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(0, change))
            losses.append(max(0, -change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        if len(closes) < slow + signal:
            return None
        
        def ema(data, period):
            if len(data) < period:
                return None
            k = 2 / (period + 1)
            ema_val = sum(data[:period]) / period
            for price in data[period:]:
                ema_val = price * k + ema_val * (1 - k)
            return ema_val
        
        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # Calculate MACD history for histogram
        macd_history = []
        for i in range(slow, len(closes) + 1):
            ef = ema(closes[:i], fast)
            es = ema(closes[:i], slow)
            if ef and es:
                macd_history.append(ef - es)
        
        signal_line = ema(macd_history, signal) if len(macd_history) >= signal else 0
        histogram = macd_line - signal_line if signal_line else 0
        
        # Histogram delta (is it expanding?)
        prev_hist = 0
        if len(macd_history) >= 2:
            prev_signal = ema(macd_history[:-1], signal) if len(macd_history) > signal else 0
            prev_hist = macd_history[-2] - prev_signal if prev_signal else 0
        
        hist_delta = histogram - prev_hist
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "hist": histogram,
            "hist_delta": hist_delta
        }
    
    @staticmethod
    def compute_heiken_ashi(candles: List[Candle]):
        if not candles:
            return []
        
        ha = []
        for i, c in enumerate(candles):
            if i == 0:
                ha_open = (c.open + c.close) / 2
                ha_close = (c.open + c.high + c.low + c.close) / 4
            else:
                ha_open = (ha[-1]["open"] + ha[-1]["close"]) / 2
                ha_close = (c.open + c.high + c.low + c.close) / 4
            
            ha.append({
                "open": ha_open,
                "close": ha_close,
                "color": "green" if ha_close > ha_open else "red"
            })
        
        return ha
    
    @staticmethod
    def count_consecutive(ha: List[dict]) -> dict:
        if not ha:
            return {"color": None, "count": 0}
        
        last_color = ha[-1]["color"]
        count = 0
        for candle in reversed(ha):
            if candle["color"] == last_color:
                count += 1
            else:
                break
        
        return {"color": last_color, "count": count}


class BTCAutoTrader:
    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0)
        self.clob_client = None
        self.total_exposure = 0.0
        self.trades_executed = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.running = False
        self.current_market = None
        # Track ALL trades for current round
        self.round_trades = []  # List of {"side": str, "bet": float, "price": float}
        
    async def initialize(self) -> bool:
        console.print(Panel.fit(
            "[bold blue]🎯 BTC 15m Auto-Trader (TA Strategy)[/bold blue]\n"
            f"Mode: {'[yellow]PAPER TRADING[/yellow]' if PAPER_TRADING else '[red]LIVE TRADING[/red]'}\n"
            f"Min Confidence: {MIN_CONFIDENCE*100:.0f}%\n"
            f"Max Bet: ${MAX_BET_SIZE}\n"
            f"Strategy: Multi-indicator confirmation"
        ))
        
        if not PRIVATE_KEY or PRIVATE_KEY == "0x-your-private-key-here":
            console.print("[red]❌ Error: Please set PRIVATE_KEY in .env[/red]")
            return False
        
        if not PAPER_TRADING:
            try:
                from py_clob_client.client import ClobClient
                self.clob_client = ClobClient(host=CLOB_API_URL, key=PRIVATE_KEY, chain_id=137)
                self.clob_client.set_api_creds(self.clob_client.derive_api_key())
                console.print("[green]✅ Connected to Polymarket[/green]")
            except Exception as e:
                console.print(f"[red]❌ Connection failed: {e}[/red]")
                return False
        else:
            console.print("[yellow]📝 Paper trading mode[/yellow]")
        
        return True
    
    async def fetch_binance_klines(self, interval: str = "1m", limit: int = 100) -> List[Candle]:
        """Fetch BTC price candles from Binance"""
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
        """Fetch current BTC 15m market"""
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
                    except:
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
            
            # Get live prices
            up_price, down_price = await self._fetch_prices(up_token, down_token)
            if up_price is None:
                up_price = float(outcome_prices[up_idx]) * 100 if up_idx < len(outcome_prices) else 50
            if down_price is None:
                down_price = float(outcome_prices[down_idx]) * 100 if down_idx < len(outcome_prices) else 50
            
            time_left = (best_end - now).total_seconds() / 60
            phase = "EARLY" if time_left > 10 else "MID" if time_left > 5 else "LATE"
            
            return BTCMarket(
                question=best_market.get("question", "")[:60],
                end_time=best_end,
                up_price=up_price,
                down_price=down_price,
                up_token_id=up_token,
                down_token_id=down_token,
                time_left_min=time_left,
                phase=phase
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
        except:
            return None, None
    
    def analyze_ta(self, candles: List[Candle], market: BTCMarket) -> TASignals:
        """Calculate all TA indicators and determine trade signal"""
        closes = [c.close for c in candles]
        
        # RSI
        rsi = TAEngine.compute_rsi(closes, 14) or 50
        if rsi >= 55:
            rsi_signal = "LONG"
        elif rsi <= 45:
            rsi_signal = "SHORT"
        else:
            rsi_signal = "NEUTRAL"
        
        # MACD
        macd = TAEngine.compute_macd(closes)
        if macd:
            if macd["hist"] > 0:
                macd_signal = "bullish"
            elif macd["hist"] < 0:
                macd_signal = "bearish"
            else:
                macd_signal = "neutral"
            macd_expanding = (macd["hist"] > 0 and macd["hist_delta"] > 0) or (macd["hist"] < 0 and macd["hist_delta"] < 0)
        else:
            macd_signal = "neutral"
            macd_expanding = False
        
        # Heiken Ashi
        ha = TAEngine.compute_heiken_ashi(candles)
        consec = TAEngine.count_consecutive(ha)
        heiken_color = consec["color"] or "neutral"
        heiken_count = consec["count"]
        
        # Delta 1m and 3m
        delta_1m = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        delta_3m = closes[-1] - closes[-4] if len(closes) >= 4 else 0
        
        if delta_1m > 0 and delta_3m > 0:
            delta_signal = "LONG"
        elif delta_1m < 0 and delta_3m < 0:
            delta_signal = "SHORT"
        else:
            delta_signal = "NEUTRAL"
        
        # Count aligned indicators
        long_count = 0
        short_count = 0
        
        # RSI
        if rsi_signal == "LONG":
            long_count += 1
        elif rsi_signal == "SHORT":
            short_count += 1
        
        # MACD
        if macd_signal == "bullish":
            long_count += 1
        elif macd_signal == "bearish":
            short_count += 1
        
        # Heiken Ashi (needs 2+ consecutive)
        if heiken_color == "green" and heiken_count >= 2:
            long_count += 1
        elif heiken_color == "red" and heiken_count >= 2:
            short_count += 1
        
        # Delta
        if delta_signal == "LONG":
            long_count += 1
        elif delta_signal == "SHORT":
            short_count += 1
        
        # Market price bias (what does the market think?)
        total = market.up_price + market.down_price
        market_up_prob = market.up_price / total if total > 0 else 0.5
        if market_up_prob > 0.55:
            long_count += 1
        elif market_up_prob < 0.45:
            short_count += 1
        
        # Determine direction
        if long_count >= 3 and long_count > short_count:
            direction = "LONG"
            confidence = 0.5 + (long_count * 0.1)
        elif short_count >= 3 and short_count > long_count:
            direction = "SHORT"
            confidence = 0.5 + (short_count * 0.1)
        else:
            direction = "NEUTRAL"
            confidence = 0.5
        
        # Rule 2: Extreme RSI = likely reversal
        if rsi < 25 or rsi > 75:
            # Don't chase extreme RSI
            if (rsi > 75 and direction == "LONG") or (rsi < 25 and direction == "SHORT"):
                direction = "NEUTRAL"
                confidence = 0.4
        
        # Trade signal: need 3+ indicators aligned and confidence >= 60%
        indicators_aligned = max(long_count, short_count)
        trade_signal = indicators_aligned >= 3 and confidence >= MIN_CONFIDENCE and direction != "NEUTRAL"
        
        # Rule 4: Entry timing
        # 5-10 min = best (MID phase)
        # Last 5 min = risky
        if market.phase == "LATE" and market.time_left_min < 2:
            trade_signal = False  # Too risky near end
        
        return TASignals(
            heiken_color=heiken_color,
            heiken_count=heiken_count,
            rsi=rsi,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            macd_expanding=macd_expanding,
            delta_1m=delta_1m,
            delta_3m=delta_3m,
            delta_signal=delta_signal,
            direction=direction,
            confidence=min(confidence, 1.0),
            indicators_aligned=indicators_aligned,
            trade_signal=trade_signal
        )
    
    def display_ta(self, ta: TASignals, market: BTCMarket):
        """Display TA analysis"""
        table = Table(title="Technical Analysis", show_header=False, box=None)
        table.add_column("Indicator", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Signal", style="bold")
        
        # Heiken
        heiken_color = "[green]" if ta.heiken_color == "green" else "[red]"
        table.add_row("Heiken Ashi", f"{heiken_color}{ta.heiken_color} x{ta.heiken_count}[/]", 
                     "[green]LONG[/]" if ta.heiken_color == "green" and ta.heiken_count >= 2 else 
                     "[red]SHORT[/]" if ta.heiken_color == "red" and ta.heiken_count >= 2 else "[dim]WEAK[/]")
        
        # RSI
        rsi_color = "[green]" if ta.rsi > 55 else "[red]" if ta.rsi < 45 else "[yellow]"
        table.add_row("RSI", f"{rsi_color}{ta.rsi:.1f}[/]", 
                     f"[green]{ta.rsi_signal}[/]" if ta.rsi_signal == "LONG" else 
                     f"[red]{ta.rsi_signal}[/]" if ta.rsi_signal == "SHORT" else "[dim]NEUTRAL[/]")
        
        # MACD
        macd_color = "[green]" if ta.macd_signal == "bullish" else "[red]" if ta.macd_signal == "bearish" else "[yellow]"
        table.add_row("MACD", f"{macd_color}{ta.macd_signal}{' (expanding)' if ta.macd_expanding else ''}[/]",
                     "[green]LONG[/]" if ta.macd_signal == "bullish" else 
                     "[red]SHORT[/]" if ta.macd_signal == "bearish" else "[dim]NEUTRAL[/]")
        
        # Delta
        delta_str = f"1m: {'+' if ta.delta_1m > 0 else ''}{ta.delta_1m:.0f} | 3m: {'+' if ta.delta_3m > 0 else ''}{ta.delta_3m:.0f}"
        table.add_row("Delta 1/3m", delta_str,
                     f"[green]{ta.delta_signal}[/]" if ta.delta_signal == "LONG" else 
                     f"[red]{ta.delta_signal}[/]" if ta.delta_signal == "SHORT" else "[dim]NEUTRAL[/]")
        
        console.print(table)
        
        # Summary
        dir_color = "[green]" if ta.direction == "LONG" else "[red]" if ta.direction == "SHORT" else "[yellow]"
        console.print(f"\n{dir_color}Prediction: {ta.direction} ({ta.confidence*100:.0f}% conf)[/]")
        console.print(f"Indicators aligned: {ta.indicators_aligned}/5")
        
        if ta.trade_signal:
            console.print(f"[bold green]✅ TRADE SIGNAL: {ta.direction}[/bold green]")
        else:
            reasons = []
            if ta.indicators_aligned < 3:
                reasons.append(f"Need 3+ indicators aligned (have {ta.indicators_aligned})")
            if ta.confidence < MIN_CONFIDENCE:
                reasons.append(f"Confidence {ta.confidence*100:.0f}% < {MIN_CONFIDENCE*100:.0f}%")
            if ta.direction == "NEUTRAL":
                reasons.append("No clear direction")
            console.print(f"[dim]No trade: {', '.join(reasons)}[/dim]")
    
    async def execute_trade(self, market: BTCMarket, direction: str) -> bool:
        """Execute trade"""
        # LONG = buy UP, SHORT = buy DOWN
        if direction == "LONG":
            token_id = market.up_token_id
            price = market.up_price / 100
            side = "UP"
        else:
            token_id = market.down_token_id
            price = market.down_price / 100
            side = "DOWN"
        
        bet = min(MAX_BET_SIZE, MAX_TOTAL_EXPOSURE - self.total_exposure)
        
        if bet <= 0:
            console.print("[yellow]Max exposure reached[/yellow]")
            return False
        
        # Store trade in round_trades list
        self.round_trades.append({
            "side": side,
            "bet": bet,
            "price": price
        })
        self.current_market = market.question
        
        if PAPER_TRADING:
            console.print(f"[yellow][PAPER] BUY {side} ${bet:.2f} @ {price*100:.1f}¢[/yellow]")
            self.trades_executed += 1
            self.total_exposure += bet
            return True
        else:
            try:
                from py_clob_client.clob_types import OrderType
                shares = bet / price
                order = self.clob_client.create_and_post_order(
                    token_id=token_id, side="BUY", size=shares, price=price, order_type=OrderType.FOK
                )
                if order and order.get("status") != "REJECTED":
                    self.trades_executed += 1
                    self.total_exposure += bet
                    console.print(f"[green]✅ Bought {side} ${bet:.2f}[/green]")
                    return True
                return False
            except Exception as e:
                logger.error(f"Trade error: {e}")
                return False
    
    def is_trading_hours(self) -> bool:
        """Check if current time is within trading window"""
        if not TRADE_START_TIME or not TRADE_END_TIME:
            return True  # No window set = trade 24/7
        
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            
            # Simple string comparison works for HH:MM format
            if TRADE_START_TIME <= TRADE_END_TIME:
                # Normal case: e.g., 09:00 to 21:00
                return TRADE_START_TIME <= current_time <= TRADE_END_TIME
            else:
                # Overnight case: e.g., 21:00 to 09:00
                return current_time >= TRADE_START_TIME or current_time <= TRADE_END_TIME
        except:
            return True
    
    def settle_round(self):
        """Settle all trades when market ends and calculate profit"""
        if not self.round_trades:
            return
        
        console.print(f"\n[bold cyan]{'═'*40}[/bold cyan]")
        console.print(f"[bold cyan]       ROUND SETTLED - {len(self.round_trades)} TRADES[/bold cyan]")
        console.print(f"[bold cyan]{'═'*40}[/bold cyan]")
        
        total_bet = 0
        total_win_payout = 0
        
        for i, trade in enumerate(self.round_trades, 1):
            bet = trade["bet"]
            price = trade["price"]
            side = trade["side"]
            shares = bet / price
            win_payout = shares * 1.0
            
            total_bet += bet
            total_win_payout += win_payout
            
            console.print(f"[dim]#{i}[/dim] {side} @ {price*100:.0f}¢ | ${bet:.2f} → {shares:.2f} shares")
        
        total_win_profit = total_win_payout - total_bet
        
        console.print(f"\n[bold]TOTAL BET: ${total_bet:.2f}[/bold]")
        console.print(f"[green]IF WIN: Payout ${total_win_payout:.2f}, Profit +${total_win_profit:.2f}[/green]")
        console.print(f"[red]IF LOSE: Loss -${total_bet:.2f}[/red]")
        console.print(f"[dim](Check Polymarket for actual result)[/dim]")
        console.print(f"[bold cyan]{'═'*40}[/bold cyan]\n")
        
        # Reset for next round
        self.total_exposure = 0
        self.round_trades = []
        self.current_market = None
    
    def display_stats(self):
        """Display running statistics"""
        profit_color = "[green]" if self.total_profit >= 0 else "[red]"
        profit_sign = "+" if self.total_profit >= 0 else ""
        console.print(f"\n[dim]Trades: {self.trades_executed} | W: {self.wins} L: {self.losses} | Exposure: ${self.total_exposure:.2f} | P/L: {profit_color}{profit_sign}${self.total_profit:.2f}[/][/dim]")
    
    async def run(self):
        if not await self.initialize():
            return
        
        self.running = True
        console.print("\n[bold green]🚀 Started with TA Strategy![/bold green]")
        console.print(f"[dim]Trade window: {TRADE_WINDOW_START:.0f} to {TRADE_WINDOW_END:.0f} min left[/dim]")
        console.print("[dim]Rules: 3+ indicators aligned, 60%+ confidence[/dim]\n")
        
        try:
            while self.running:
                console.print(f"\n{'='*50}")
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")
                
                # Fetch market and candles
                market = await self.fetch_btc_market()
                candles = await self.fetch_binance_klines("1m", 100)
                
                if not market:
                    console.print("[yellow]No active market[/yellow]")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                
                if not candles:
                    console.print("[yellow]No candle data[/yellow]")
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                
                # Check if market changed (new round) - settle previous trades
                if self.current_market and self.current_market != market.question:
                    self.settle_round()
                
                # Display market
                console.print(f"Market: {market.question}")
                console.print(f"Time Left: [bold]{market.time_left_min:.1f} min[/bold] ({market.phase})")
                console.print(f"Prices: [green]UP {market.up_price:.0f}¢[/green] | [red]DOWN {market.down_price:.0f}¢[/red]\n")
                
                # Check if within trade window
                in_window = TRADE_WINDOW_END <= market.time_left_min <= TRADE_WINDOW_START
                
                if not in_window:
                    console.print(f"[yellow]Outside trade window ({TRADE_WINDOW_START:.0f}-{TRADE_WINDOW_END:.0f} min left)[/yellow]")
                    self.display_stats()
                    await asyncio.sleep(SCAN_INTERVAL)
                    continue
                
                # Analyze TA
                ta = self.analyze_ta(candles, market)
                self.display_ta(ta, market)
                
                # Execute trade if signal
                if ta.trade_signal and self.total_exposure < MAX_TOTAL_EXPOSURE:
                    await self.execute_trade(market, ta.direction)
                
                self.display_stats()
                await asyncio.sleep(SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped[/yellow]")
        finally:
            await self.http.aclose()


async def main():
    trader = BTCAutoTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
