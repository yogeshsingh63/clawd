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

# TA config aligned with btc_assistant
CANDLE_WINDOW_MINUTES = int(os.getenv("CANDLE_WINDOW_MINUTES", "15"))
VWAP_SLOPE_LOOKBACK = int(os.getenv("VWAP_SLOPE_LOOKBACK", "5"))

# Aggressive risk tuning (Option 1)
AGGRESSIVE_MODE = os.getenv("AGGRESSIVE_MODE", "true").lower() == "true"
MID_MIN_PROB = float(os.getenv("MID_MIN_PROB", "0.55"))  # was 0.60
MID_EDGE_THRESHOLD = float(os.getenv("MID_EDGE_THRESHOLD", "0.05"))  # was 0.10
MOMENTUM_OVERRIDE_MIN = int(os.getenv("MOMENTUM_OVERRIDE_MIN", "4"))  # 4 of 5 signals

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
    rsi: Optional[float]
    rsi_slope: Optional[float]
    rsi_signal: str  # "LONG", "SHORT", "NEUTRAL"
    
    # MACD
    macd_signal: str  # "bullish", "bearish", "neutral"
    macd_expanding: bool
    
    # Delta
    delta_1m: float
    delta_3m: float
    delta_signal: str  # "LONG", "SHORT", "NEUTRAL"

    # VWAP
    vwap: Optional[float]
    vwap_slope: Optional[float]
    vwap_dist: Optional[float]

    # Model + Edge
    model_up: Optional[float]
    model_down: Optional[float]
    edge_up: Optional[float]
    edge_down: Optional[float]
    up_score: float
    down_score: float
    rec_action: str
    rec_side: Optional[str]
    rec_phase: str
    rec_reason: Optional[str]
    
    # Overall
    direction: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float
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
    def sma(values: List[float], period: int) -> Optional[float]:
        if len(values) < period:
            return None
        slice_vals = values[-period:]
        return sum(slice_vals) / period

    @staticmethod
    def slope_last(values: List[float], points: int = 3) -> Optional[float]:
        if len(values) < points:
            return None
        slice_vals = values[-points:]
        first = slice_vals[0]
        last = slice_vals[-1]
        return (last - first) / (points - 1)

    @staticmethod
    def compute_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        if len(closes) < slow + signal:
            return None

        def ema(data, period):
            if len(data) < period:
                return None
            k = 2 / (period + 1)
            prev = data[0]
            for price in data[1:]:
                prev = price * k + prev * (1 - k)
            return prev

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)

        if ema_fast is None or ema_slow is None:
            return None

        macd_line = ema_fast - ema_slow

        macd_series = []
        for i in range(1, len(closes) + 1):
            sub = closes[:i]
            ef = ema(sub, fast)
            es = ema(sub, slow)
            if ef is None or es is None:
                continue
            macd_series.append(ef - es)

        signal_line = ema(macd_series, signal)
        if signal_line is None:
            return None

        hist = macd_line - signal_line
        prev_hist = None
        if len(macd_series) >= signal + 1:
            prev_signal = ema(macd_series[:-1], signal)
            if prev_signal is not None:
                prev_hist = macd_series[-2] - prev_signal

        return {
            "macd": macd_line,
            "signal": signal_line,
            "hist": hist,
            "hist_delta": None if prev_hist is None else (hist - prev_hist)
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

    @staticmethod
    def compute_session_vwap(candles: List[Candle]) -> Optional[float]:
        if not candles:
            return None
        pv = 0.0
        v = 0.0
        for c in candles:
            tp = (c.high + c.low + c.close) / 3
            pv += tp * c.volume
            v += c.volume
        if v == 0:
            return None
        return pv / v

    @staticmethod
    def compute_vwap_series(candles: List[Candle]) -> List[Optional[float]]:
        series = []
        pv = 0.0
        v = 0.0
        for c in candles:
            tp = (c.high + c.low + c.close) / 3
            pv += tp * c.volume
            v += c.volume
            series.append(None if v == 0 else pv / v)
        return series

    @staticmethod
    def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, val))

    @staticmethod
    def score_direction(
        price: Optional[float],
        vwap: Optional[float],
        vwap_slope: Optional[float],
        rsi: Optional[float],
        rsi_slope: Optional[float],
        macd: Optional[dict],
        heiken_color: Optional[str],
        heiken_count: int,
        failed_vwap_reclaim: bool,
    ):
        up = 1.0
        down = 1.0

        if price is not None and vwap is not None:
            if price > vwap:
                up += 2
            if price < vwap:
                down += 2

        if vwap_slope is not None:
            if vwap_slope > 0:
                up += 2
            if vwap_slope < 0:
                down += 2

        if rsi is not None and rsi_slope is not None:
            if rsi > 55 and rsi_slope > 0:
                up += 2
            if rsi < 45 and rsi_slope < 0:
                down += 2

        if macd and macd.get("hist") is not None and macd.get("hist_delta") is not None:
            expanding_green = macd["hist"] > 0 and macd["hist_delta"] > 0
            expanding_red = macd["hist"] < 0 and macd["hist_delta"] < 0
            if expanding_green:
                up += 2
            if expanding_red:
                down += 2

            if macd.get("macd") is not None:
                if macd["macd"] > 0:
                    up += 1
                if macd["macd"] < 0:
                    down += 1

        if heiken_color:
            if heiken_color == "green" and heiken_count >= 2:
                up += 1
            if heiken_color == "red" and heiken_count >= 2:
                down += 1

        if failed_vwap_reclaim:
            down += 3

        raw_up = up / (up + down)
        return {"up_score": up, "down_score": down, "raw_up": raw_up}

    @staticmethod
    def apply_time_awareness(raw_up: float, remaining_minutes: float, window_minutes: int):
        time_decay = TAEngine.clamp(remaining_minutes / window_minutes, 0.0, 1.0)
        adjusted_up = TAEngine.clamp(0.5 + (raw_up - 0.5) * time_decay, 0.0, 1.0)
        return {"time_decay": time_decay, "adjusted_up": adjusted_up, "adjusted_down": 1 - adjusted_up}

    @staticmethod
    def compute_edge(model_up: Optional[float], model_down: Optional[float], market_yes: Optional[float], market_no: Optional[float]):
        if market_yes is None or market_no is None:
            return {"market_up": None, "market_down": None, "edge_up": None, "edge_down": None}
        total = market_yes + market_no
        market_up = (market_yes / total) if total > 0 else None
        market_down = (market_no / total) if total > 0 else None
        edge_up = None if market_up is None else model_up - market_up
        edge_down = None if market_down is None else model_down - market_down
        return {
            "market_up": None if market_up is None else TAEngine.clamp(market_up, 0.0, 1.0),
            "market_down": None if market_down is None else TAEngine.clamp(market_down, 0.0, 1.0),
            "edge_up": edge_up,
            "edge_down": edge_down,
        }

    @staticmethod
    def decide(
        remaining_minutes: float,
        edge_up: Optional[float],
        edge_down: Optional[float],
        model_up: Optional[float],
        model_down: Optional[float],
        aggressive: bool = False,
        momentum_side: Optional[str] = None,
        momentum_score: int = 0,
        momentum_min: int = 4,
    ):
        phase = "EARLY" if remaining_minutes > 10 else "MID" if remaining_minutes > 5 else "LATE"
        threshold = 0.05 if phase == "EARLY" else 0.1 if phase == "MID" else 0.2
        min_prob = 0.55 if phase == "EARLY" else 0.6 if phase == "MID" else 0.65

        if aggressive and phase == "MID":
            threshold = MID_EDGE_THRESHOLD
            min_prob = MID_MIN_PROB

        if edge_up is None or edge_down is None:
            return {"action": "NO_TRADE", "side": None, "phase": phase, "reason": "missing_market_data"}

        best_side = "UP" if edge_up > edge_down else "DOWN"
        best_edge = edge_up if best_side == "UP" else edge_down
        best_model = model_up if best_side == "UP" else model_down

        # Momentum override: allow entry on strong directional alignment
        if aggressive and momentum_side in ("UP", "DOWN") and momentum_score >= momentum_min:
            override_model = model_up if momentum_side == "UP" else model_down
            if override_model is None or override_model >= 0.5:
                return {"action": "ENTER", "side": momentum_side, "phase": phase, "strength": "MOMENTUM", "edge": best_edge}

        if best_edge < threshold:
            return {"action": "NO_TRADE", "side": None, "phase": phase, "reason": f"edge_below_{threshold}"}

        if best_model is not None and best_model < min_prob:
            return {"action": "NO_TRADE", "side": None, "phase": phase, "reason": f"prob_below_{min_prob}"}

        strength = "STRONG" if best_edge >= 0.2 else "GOOD" if best_edge >= 0.1 else "OPTIONAL"
        return {"action": "ENTER", "side": best_side, "phase": phase, "strength": strength, "edge": best_edge}


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
            f"Decision: Edge + model prob thresholds (btc_assistant)\n"
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
        last_price = closes[-1] if closes else None
        
        # RSI + slope (align with btc_assistant)
        rsi = TAEngine.compute_rsi(closes, 14)
        rsi_series = []
        for i in range(len(closes)):
            sub = closes[: i + 1]
            r = TAEngine.compute_rsi(sub, 14)
            if r is not None:
                rsi_series.append(r)
        rsi_slope = TAEngine.slope_last(rsi_series, 3)
        if rsi_slope is not None:
            rsi_signal = "LONG" if rsi_slope > 0 else "SHORT" if rsi_slope < 0 else "NEUTRAL"
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
            macd_expanding = macd.get("hist_delta") is not None and (
                (macd["hist"] > 0 and macd["hist_delta"] > 0) or (macd["hist"] < 0 and macd["hist_delta"] < 0)
            )
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

        # VWAP + slope
        vwap_series = TAEngine.compute_vwap_series(candles)
        vwap = vwap_series[-1] if vwap_series else None
        if vwap_series and len(vwap_series) >= VWAP_SLOPE_LOOKBACK:
            vwap_slope = (vwap_series[-1] - vwap_series[-VWAP_SLOPE_LOOKBACK]) / VWAP_SLOPE_LOOKBACK if vwap_series[-1] is not None and vwap_series[-VWAP_SLOPE_LOOKBACK] is not None else None
        else:
            vwap_slope = None
        vwap_dist = ((last_price - vwap) / vwap) if (last_price is not None and vwap is not None and vwap != 0) else None

        failed_vwap_reclaim = False
        if vwap_series and len(vwap_series) >= 3 and vwap_series[-1] is not None and vwap_series[-2] is not None:
            failed_vwap_reclaim = closes[-1] < vwap_series[-1] and closes[-2] > vwap_series[-2]

        # Momentum alignment (5 signals)
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
            momentum_side = "UP"
            momentum_score = momentum_long
        elif momentum_short > momentum_long:
            momentum_side = "DOWN"
            momentum_score = momentum_short
        else:
            momentum_side = None
            momentum_score = 0

        # Score model direction (btc_assistant logic)
        scored = TAEngine.score_direction(
            price=last_price,
            vwap=vwap,
            vwap_slope=vwap_slope,
            rsi=rsi,
            rsi_slope=rsi_slope,
            macd=macd,
            heiken_color=heiken_color,
            heiken_count=heiken_count,
            failed_vwap_reclaim=failed_vwap_reclaim,
        )
        time_aware = TAEngine.apply_time_awareness(scored["raw_up"], market.time_left_min, CANDLE_WINDOW_MINUTES)
        model_up = time_aware["adjusted_up"]
        model_down = time_aware["adjusted_down"]

        # Edge vs market prices (Polymarket)
        edge = TAEngine.compute_edge(model_up, model_down, market.up_price, market.down_price)
        rec = TAEngine.decide(
            market.time_left_min,
            edge["edge_up"],
            edge["edge_down"],
            model_up,
            model_down,
            aggressive=AGGRESSIVE_MODE,
            momentum_side=momentum_side,
            momentum_score=momentum_score,
            momentum_min=MOMENTUM_OVERRIDE_MIN,
        )

        if rec["action"] == "ENTER":
            direction = "LONG" if rec["side"] == "UP" else "SHORT"
            confidence = model_up if rec["side"] == "UP" else model_down
        else:
            direction = "NEUTRAL"
            confidence = 0.0

        trade_signal = rec["action"] == "ENTER"
        
        return TASignals(
            heiken_color=heiken_color,
            heiken_count=heiken_count,
            rsi=rsi,
            rsi_slope=rsi_slope,
            rsi_signal=rsi_signal,
            macd_signal=macd_signal,
            macd_expanding=macd_expanding,
            delta_1m=delta_1m,
            delta_3m=delta_3m,
            delta_signal=delta_signal,
            vwap=vwap,
            vwap_slope=vwap_slope,
            vwap_dist=vwap_dist,
            model_up=model_up,
            model_down=model_down,
            edge_up=edge["edge_up"],
            edge_down=edge["edge_down"],
            up_score=scored["up_score"],
            down_score=scored["down_score"],
            rec_action=rec["action"],
            rec_side=rec.get("side"),
            rec_phase=rec["phase"],
            rec_reason=rec.get("reason"),
            direction=direction,
            confidence=min(confidence, 1.0) if confidence is not None else 0.0,
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
        rsi_color = "[green]" if ta.rsi_signal == "LONG" else "[red]" if ta.rsi_signal == "SHORT" else "[yellow]"
        rsi_value = "-" if ta.rsi is None else f"{ta.rsi:.1f}"
        rsi_arrow = "↑" if ta.rsi_slope and ta.rsi_slope > 0 else "↓" if ta.rsi_slope and ta.rsi_slope < 0 else "-"
        table.add_row("RSI", f"{rsi_color}{rsi_value} {rsi_arrow}[/]", 
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

        # VWAP
        if ta.vwap is not None and ta.vwap_dist is not None:
            vwap_signal = "LONG" if ta.vwap_dist > 0 else "SHORT" if ta.vwap_dist < 0 else "NEUTRAL"
            vwap_color = "[green]" if vwap_signal == "LONG" else "[red]" if vwap_signal == "SHORT" else "[yellow]"
            vwap_dist_pct = ta.vwap_dist * 100
            table.add_row("VWAP", f"{vwap_color}{ta.vwap:.0f} ({vwap_dist_pct:+.2f}%)" + (" slope↑" if ta.vwap_slope and ta.vwap_slope > 0 else " slope↓" if ta.vwap_slope and ta.vwap_slope < 0 else " slope-") + "[/]",
                         f"[green]{vwap_signal}[/]" if vwap_signal == "LONG" else f"[red]{vwap_signal}[/]" if vwap_signal == "SHORT" else "[dim]NEUTRAL[/]")
        
        console.print(table)
        
        # Summary
        dir_color = "[green]" if ta.direction == "LONG" else "[red]" if ta.direction == "SHORT" else "[yellow]"
        conf_pct = ta.confidence * 100 if ta.confidence is not None else 0.0
        console.print(f"\n{dir_color}Model: {ta.direction} ({conf_pct:.0f}% prob)[/]")

        if ta.model_up is not None and ta.model_down is not None:
            console.print(f"[dim]Model probs: UP {ta.model_up*100:.0f}% | DOWN {ta.model_down*100:.0f}%[/dim]")
        if ta.edge_up is not None and ta.edge_down is not None:
            console.print(f"[dim]Edge vs market: UP {ta.edge_up:+.3f} | DOWN {ta.edge_down:+.3f}[/dim]")
        console.print(f"[dim]Score: UP {ta.up_score:.1f} | DOWN {ta.down_score:.1f}[/dim]")

        if ta.trade_signal:
            console.print(f"[bold green]✅ TRADE SIGNAL: {ta.direction} ({ta.rec_phase})[/bold green]")
        else:
            reason = ta.rec_reason or "no_edge"
            console.print(f"[dim]No trade ({ta.rec_phase}): {reason}[/dim]")
    
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
        console.print("[dim]Rules: edge vs market + time-aware model prob (btc_assistant)[/dim]\n")
        
        try:
            while self.running:
                console.print(f"\n{'='*50}")
                console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]")
                
                # Fetch market and candles
                market = await self.fetch_btc_market()
                candles = await self.fetch_binance_klines("1m", 240)
                
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
