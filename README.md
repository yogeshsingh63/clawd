# Polymarket Trading Suite

Automated Polymarket trading tools with a shared BTC TA engine and multiple strategies.

## Tools

### 1. BTC Auto Trader (`btc_auto_trader.py`)
TA-driven auto trader for BTC 15m UP/DOWN markets using the same logic as the BTC assistant.

### 2. Polymarket Arb Bot (`btc_arb_bot.py`)
Strict, only‑if‑locked arbitrage across **single markets with mutually exclusive outcomes**.  
Buys YES on all outcomes only if total cost < $1 after slippage/fees.

### 3. BTC Scalp Bot (`btc_scalp_bot.py`)
Supphieros‑style: buy low‑priced shares during high volatility and flip quickly for small profits.  
Uses the same TA output table as the main auto trader.

### 4. BTC Late Conviction Bot (`btc_late_conviction_bot.py`)
Bigsibas‑style: trades only in the last 5–9 minutes, larger size, holds to resolution.  
Uses the same TA output table and round PnL format as the main auto trader.

### 5. BTC Assistant (`btc_assistant/`)
Display‑only UI for BTC 15m predictions (no trades).

---

## Quick Start

### Option 1: Easy Launcher
```bash
chmod +x start.sh
./start.sh
```

### Option 2: Run Manually

Activate venv first:
```bash
source venv/bin/activate
```

Auto trader:
```bash
python3 btc_auto_trader.py
```

Arb bot:
```bash
python3 btc_arb_bot.py
```

Scalp bot:
```bash
python3 btc_scalp_bot.py
```

Late conviction bot:
```bash
python3 btc_late_conviction_bot.py
```

BTC assistant UI:
```bash
cd btc_assistant && npm start
```

---

## Configuration (`.env`)

Common:
```
PRIVATE_KEY=0x...
PAPER_TRADING=true
LOG_LEVEL=INFO
```

Auto trader:
```
MAX_BET_SIZE=1.00
MAX_TOTAL_EXPOSURE=3
BTC_MIN_CONFIDENCE=0.60
SCAN_INTERVAL=10
TRADE_WINDOW_START=15
TRADE_WINDOW_END=0
SETTLEMENT_GRACE_MIN=3
```

Arb bot:
```
ARB_ORDER_SIZE=10
MIN_PROFIT_PCT=0.5
ARB_SLIPPAGE_BPS=5
ARB_FEE_PCT=0.0
ARB_MAX_TRADES_PER_MARKET=1
ARB_MIN_OUTCOMES=2
ARB_MIN_LIQUIDITY=0
ARB_MAX_MARKETS_SCAN=200
ARB_MARKETS_PER_CYCLE=40
ARB_CONCURRENCY=5
ARB_PREFILTER_MAX_SUM=1.0
```

Scalp bot:
```
SCALP_ORDER_SIZE=50
SCALP_MAX_ENTRY_PRICE=0.30
SCALP_TAKE_PROFIT=0.02
SCALP_STOP_LOSS=0.02
SCALP_MIN_ALIGN=3
SCALP_MIN_BTC_MOVE_PCT=0.05
SCALP_MAX_HOLD_SEC=180
SCALP_FORCE_EXIT_MIN=1.5
SCALP_MAX_SIMUL_POSITIONS=2
SCALP_SCAN_INTERVAL=5
```

Late conviction bot:
```
LATE_ENTRY_START=9
LATE_ENTRY_END=5
LATE_MIN_PROB=0.58
LATE_ORDER_SIZE_USD=5.0
LATE_MAX_TRADES_PER_ROUND=2
LATE_SCAN_INTERVAL=10
SETTLEMENT_GRACE_MIN=3
```

---

## Files

| File | Description |
|------|-------------|
| `btc_auto_trader.py` | BTC 15m auto trader (TA + edge logic) |
| `btc_arb_bot.py` | Strict multi‑outcome arbitrage bot |
| `btc_scalp_bot.py` | Volatility scalp bot |
| `btc_late_conviction_bot.py` | Late‑window conviction bot |
| `btc_assistant/` | BTC 15m prediction UI |
| `.env` | Private config |
| `start.sh` | Launcher |
| `venv/` | Python virtual environment |
